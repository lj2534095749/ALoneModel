import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from block.cbl import CBL as BaseConv
from loss.iou_loss import IOUloss, bboxes_iou


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        """
        [
            3 * Sequential[
                CBL(i=256,o=256,k=3,s=1,p=same),
                CBL(i=256,o=256,k=3,s=1,p=same),
            ],
        ]
        """
        self.cls_convs = nn.ModuleList()

        """
        [
            3 * Sequential[
                CBL(i=256,o=256,k=3,s=1,p=same),
                CBL(i=256,o=256,k=3,s=1,p=same),
            ],
        ]
        """
        self.reg_convs = nn.ModuleList()

        """主要对目标框的类别，预测分数
        [
            CBL(i=256,o=n_anchors * num_classes,k=1,s=1,p=same),
        ]
        """
        self.cls_preds = nn.ModuleList()

        """主要对目标框的坐标信息（x,y,w,h）进行预测
        [
            CBL(i=256,o=4,k=1,s=1,p=same),
        ]
        """
        self.reg_preds = nn.ModuleList()

        """主要判断目标框是前景还是背景
        [
            CBL(i=256,o=n_anchors,k=1,s=1,p=same),
        ]
        """
        self.obj_preds = nn.ModuleList()

        """
        [
            CBL(i=c[0],o=256,k=1,s=1,p=same),
            CBL(i=c[1],o=256,k=1,s=1,p=same),
            CBL(i=c[2],o=256,k=1,s=1,p=same),
        ]
        """
        self.stems = nn.ModuleList()

        # Conv = DWConv if depthwise else BaseConv
        Conv = BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        # [tensor([0.]), tensor([0.]), tensor([0.])]
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # b,256,w,h
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            # b,256,w,h
            cls_feat = cls_conv(cls_x)
            # b,a*n,w,h
            cls_output = self.cls_preds[k](cls_feat)

            # b,256,w,h
            reg_feat = reg_conv(reg_x)
            # b,4,w,h
            reg_output = self.reg_preds[k](reg_feat)
            # b,a,w,h
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                # b,4+a+a*n,w,h
                output = torch.cat([reg_output, obj_output, cls_output], dim=1)
                # output(b,a*h*w,n_ch) grid(1,1*h*w,2)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                # 1,1*h*w
                x_shifts.append(grid[:, :, 0])  # 获得第k个检测头每个网格点在x方向上的偏移
                y_shifts.append(grid[:, :, 1])  # 获得第k个检测头每个网格点在y方向上的偏移
                # 1,1*h*w
                expanded_strides.append(  # 表示每个网格的步长，维度和x_shifts相同，每个元素值为stride_this_level
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                # b,4+a+a*n,w,h
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        a_ = a*n+a+4 = anchor_attr
        :param output: 当前检测头的输出，维度为(batch_size, anchor_attr, grid_w, grid_h)
        :param k: 当前检测头的序号，如果只有三个检测头，那么k=0~2
        :param stride: 当前检测头输出的特征层的步长
        :param dtype:
        :return: output:预测框数据调整后的结果，已将预测框的中心点坐标和高宽转化成letterbox中的数据，
                        维度为(batch_size, anchor_attr, grid_w, grid_h)
                 grid:表示每个网格左上角点在输出特征图中的位置
        """
        grid = self.grids[k]  # 第k个检测头的grid

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]  # 特征图的高宽
        if grid.shape[2:4] != output.shape[2:4]:  # 最开始的时候，grid是tensor([0.])，自然能进入循环
            """torch.meshgrid(): 生成网格，可以用于生成坐标
            # 函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，
            # 当两个输入张量数据类型不同或维度不是一维时会报错
            # 其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同
            """
            # yv(h,w) xv(h,w)
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # yv(h,w) xv(h,w) -> grid(h, w, 2) -> grid(1, 1, h, w, 2)
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid  # 更新第k个检测头对应的grid

        # b,a_,w,h -> b,a,n_ch,h,w
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        # b,a,n_ch,h,w -> b,a,h,w,n_ch -> b,a*h*w,n_ch
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        # 1,1,h,w,2 -> 1,1*h*w,2
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride  # 将预测框中心点坐标转化为letterbox图像中的坐标
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # 将预测框的宽高转化为letterbox图像中的宽高
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        """计算当前batch的损失，以下注释都是假设 batch_size=b，input_size=(640, 640)，num_cls=n 的情况下的注释
        n_anchors_all = 6400 + 1600 + 400 = 8400
        :param imgs:
        :param x_shifts: 列表，对应检测头的各个网格在x方向上的偏移，三个元素的维度分别为：(1, 6400), (1, 1600), (1, 400)
        :param y_shifts: 列表，对应检测头的各个网格在y方向上的偏移，三个元素的维度分别为：(1, 6400), (1, 1600), (1, 400)
        :param expanded_strides: 列表，每个元素的维度和x_shifts对应的元素相同，表示每个网格（anchor）的步长
        :param labels: 列表中的每个元素为(b, num_gt, 5)，(b, num_gt, class_index x y w h)
        :param outputs: 模型输出后经过torch.cat的结果，维度为 torch.Size([b, 8400, 5+n])，16是batch_size，8400是anchor的数量
        :param origin_preds:
        :param dtype:
        :return:
        """
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4], x y w h
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1], obj
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls], class

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]  # n_anchors_all
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        # foreach batch
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])  # 当前图片的GT数目，即真实框数目
            num_gts += num_gt  # batch内图片的GT数目
            if num_gt == 0:  # 如果第batch_idx张图片中，GT的数目为0
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # GT的中心点坐标及宽高，维度为(num_gt, 4)
                gt_classes = labels[batch_idx, :num_gt, 0]  # GT的类别索引
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 预测框的中心点坐标及宽高，维度为(n_anchors_all, 4)

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                """分类目标，维度为(len_sg, 4)
                # F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes)返回的张量维度为(len_sg, 4)
                # pred_ious_this_matching.unsqueeze(-1)返回的张量维度为(len_sg, 1)
                """
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)

                """置信度目标, # 维度为(8400, 1)
                # GT的置信度是1
                # obj_target非0则1，这里之所以还有0，是因为负样本（即没有和GT匹配的anchor）也是参与置信度损失计算的
                """
                obj_target = fg_mask.unsqueeze(-1)

                """回归目标, 维度为(len_sg, 4)
                # 通常，len_sg是大于num_gt的，可以认为，在计算损失函数的时候，有些GT被使用了多次
                """
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)  # 分类标签
        reg_targets = torch.cat(reg_targets, 0)  # 回归标签
        obj_targets = torch.cat(obj_targets, 0)  # 置信度标签
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        # iou损失
        # fg_mask是布尔索引，bbox_preds.view(-1, 4)[fg_masks]是将能与GT匹配的anchor取出，然后和reg_targets求IOU损失
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        # 置信度损失
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        # 分类损失
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0  # 回归权重
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1  # 损失函数计算，这个才是真正的损失函数

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):
        """标签分配函数，初步筛选 + SimOTA
        # 将8400个anchor划分成正负样本，
        # 正样本就是能和GT进行匹配的anchor，
        # 负样本就是不能和GT进行匹配的anchor，
        # 正样本可以和GT计算分类、回归、置信度损失，负样本只能计算置信度损失。
        :param batch_idx:
        :param num_gt: 当前图片中GT的数量
        :param total_num_anchors: 三个检测头的anchor总数，纯数字，8400
        :param gt_bboxes_per_image: 当前图片中GT的中心点坐标及宽高，维度为(num_gt, 4)
        :param gt_classes: 前图片中，所有GT的类别索引，维度为(num_gt,)
        :param bboxes_preds_per_image: 当前图片预测框的中心点坐标及宽高，维度为(8400, 4)
        :param expanded_strides: 各个anchor与输入图片中网格的尺寸比例，即步长，维度为(1, 8400)
        :param x_shifts: 各个anchor在特征图中的横坐标
        :param y_shifts: 各个anchor在特征图中的纵坐标
        :param cls_preds:
        :param bbox_preds:
        :param obj_preds:
        :param labels:
        :param imgs:
        :param mode:
        :return: gt_matched_classes: 第二轮筛选后得到的anchor对应GT的索引，维度为(len_sg, )，len_sg是经过第二轮筛选后得到的anchor数量\
                 fg_mask: 第二轮筛选后得到的anchor在8400个anchor中的布尔索引，维度为(8400, )
                 pred_ious_this_matching: 第二轮筛选得到的anchor，与其对应的GT的iou，维度为(len_sg, )
                 matched_gt_inds: 第二轮筛选得到的anchor能和哪些GT匹配，维度为(len_sg, )
                 num_fg: 当前图片中，经过两轮筛选后，所有GT的正样本总数，即能与任意一个GT匹配的anchor总数，一个纯数字
        """

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # -----------------------------------------------#
        #  1. 初步筛选
        # -----------------------------------------------#

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        # -----------------------------------------------#
        #  2. SimOTA
        # （1）计算每个anchor（经过第一轮筛选后得到的anchor）与每个GT的分类损失和iou损失，然后求和得到cost矩阵（成本函数）；
        # （2）在经过第一轮筛选后得到的anchor中，为每个GT找到与其有最大IOU的10个anchor，将这10个anchor对应的IOU值求和取整，
        #    即为当前GT所匹配到的anchor数量，即dynamic_k，IOU排名前dynamic_k的anchor即为和当前GT匹配的anchor。
        # -----------------------------------------------#

        # 获得第一轮得到的anchor的边框、类别概率和置信度
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # 维度为(len_fg, 4)，len_fg是fg_mask中True的个数
        cls_preds_ = cls_preds[batch_idx][fg_mask]  # 维度为(len_fg, n)，因为是n个类别，所以第二个维度是n
        obj_preds_ = obj_preds[batch_idx][fg_mask]  # 维度为(len_fg, 1)
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]  # len_fg，纯数字，即通过第一轮筛选后的anchor数量

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # -----------------------------------------------#
        #  2.1. 计算IOU损失
        # -----------------------------------------------#

        # 计算GT和第一轮得到的anchor的iou（交并比）, 维度为(num_gt, len_fg)
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # iou（交并比）损失
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # -----------------------------------------------#
        #  2.2. 计算分类损失
        # -----------------------------------------------#

        """将正样本anchor的预测分类做成one-hot编码
        # F.one_hot(gt_classes.to(torch.int64), self.num_classes)返回的张量的维度为(num_gt, 4)
        # unsqueeze(1)返回的维度为(num_gt, 1, 4)
        # repeat(1, num_in_boxes_anchor, 1)返回的维度为(num_gt, len_fg, 4)
        # 上述命令执行后，gt_cls_per_image表示每个GT的类别one-hot编码
        """
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # 不使用混合精度加速
        with torch.cuda.amp.autocast(enabled=False):
            """为每个类别的置信度，维度为(num_gt, len_fg, 4)
            # cls_preds_.float().unsqueeze(0)返回的维度为(1, len_fg, 4)，repeat(num_gt, 1, 1)返回的维度为(num_gt, len_fg, 4)
            # obj_preds_.unsqueeze(0)返回的维度为(1, len_fg, 1)，.repeat(num_gt, 1, 1)返回的维度为(num_gt, len_fg, 1)
            # sigmoid_()对每个类别的概率做二分类
            """
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            """计算分类损失, (num_gt, len_fg)
            # F.binary_cross_entropy(A, B, reduction="none")的维度为(num_gt, len_fg, num_classes)
            # sum(-1)的维度为(num_gt, len_fg)
            # pair_wise_cls_loss表示第i个GT和第j个（经过第一轮筛选后的anchor的第j个）anchor的分类损失
            """
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

        del cls_preds_  # 释放内存

        # -----------------------------------------------#
        #  2.3. 计算成本函数
        # -----------------------------------------------#

        """计算成本函数
        # 维度为(num_gt, len_fg)
        """
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        """
        # 对8400个anchor做第一轮筛选。第一轮筛选使用了两种方法，任意一个anchor只要通过其中一种筛选方法，就可以认为其通过了第一轮筛选。
        # 第一种方法是先把网格的各个中心点坐标求出来，判断其是否在GT的内部，如果在GT的内部，那么就认为该网格对应的anchor与GT匹配。
        # 第二种方法是以每个GT的中心点为中心，生成一个边长为5的正方形，判断各个网格的中心点是否在这个网格的内部。
        :param gt_bboxes_per_image: 当前图片中，各个真实框的中心点坐标及宽高，维度为(num_gt, 4)
        :param expanded_strides: 每个网格的步长，维度为torch.Size([1, 8400])
        :param x_shifts:
        :param y_shifts:
        :param total_num_anchors:  网格点总数，纯数字，8400
        :param num_gt: 真实框总数
        :return: is_in_boxes_anchor 能通过两种方法之一的anchor的布尔索引，维度为(8400, )
                 is_in_boxes_and_center 这也是一个布尔索引，表示第一种筛选方法得到的anchor中，能通过第二种筛选方法的anchor，
                                        维度为(num_gt, len_first)，len_first是is_in_boxes_anchor中True的数量
        """

        expanded_strides_per_image = expanded_strides[0]  # 获得每个网格的步长，维度为(8400, )
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        """获得各个网格的中心点横坐标、纵坐标
        # (x_shifts[0] + 0.5) * expanded_strides_per_image的维度为(8400, ),
        # unsqueeze(0)之后为(1, 8400),
        # repeat(num_gt, 1)之后为(num_gt, 8400)
        """
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        # -------------------------------------------------------#
        # 第一种筛选方式：筛选出中心点在GT内部的网格，所对应的anchor
        # -------------------------------------------------------#

        """各个GT的上下左右边缘，(num_gt, 8400)
        # gt_bboxes_per_image_l：当前图片，每个真实框的左边缘横坐标
        # gt_bboxes_per_image_r：当前图片，每个真实框的右边缘横坐标
        # gt_bboxes_per_image_t：当前图片，每个真实框的上边缘横坐标
        # gt_bboxes_per_image_b：当前图片，每个真实框的下边缘横坐标
        """
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        """计算各个网格中心点与GT各个边缘的距离，维度为(num_gt, 8400)
        # 如果b_l>0，表示对应的网格中心在GT左边缘的右边
        # 如果b_r>0，表示对应的网格中心在GT右边缘的左边
        # 如果b_t>0，表示对应的网格中心在GT上边缘的下边
        # 如果b_b>0，表示对应的网格中心在GT下边缘的上边
        """
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # num_gt,8400,4
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        """获得各个anchor的匹配情况
        # 只有当最后一个维度的4个数都大于0，才说明对应网格的中心点在GT的内部
        # 获得一个布尔索引，维度为(num_gt, 8400)，如果is_in_boxes[i, j]为True，表示第i个GT和第j个网格对应的anchor能匹配上
        """
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0

        """获得正样本的索引
        # 计算每个网格能与多少个GT进行匹配，>0表示对应的对应的anchor至少存在一个GT与之匹配
        # 返回值的维度为(8400, )
        """
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # -------------------------------------------------------#
        # 第二种筛选方式：以GT的中心为中心，生成一个边长为5个stride的正方形（GT方框），将中心点落在这个正方形内的网格所对应的anchor，作为与GT匹配的正样本
        # -------------------------------------------------------#

        # in fixed center，GT方框的半径
        center_radius = 2.5

        """获得GT方框的左右上下边缘
        # gt_bboxes_per_image[:, 0]的维度为(num_gt, )，
        # unsqueeze(1)的维度为(num_gt, 1)，
        # repeat(1, total_num_anchors)的维度为(num_gt, 8400)
        """
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        """计算各个网格中心点与GT方框各个边缘的距离，维度为(num_gt, 8400)
        # 如果c_l>0，表示对应的网格中心在GT方框左边缘的右边
        # 如果c_r>0，表示对应的网格中心在GT方框右边缘的左边
        # 如果c_t>0，表示对应的网格中心在GT方框上边缘的下边
        # 如果c_b>0，表示对应的网格中心在GT方框下边缘的上边
        """
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        # num_gt,8400,4
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        """获得各个anchor的匹配情况
        # 只有当最后一个维度的4个数都大于0，才说明对应网格的中心点在GT的内部
        # 获得一个布尔索引，维度为(num_gt, 8400)，如果is_in_boxes[i, j]为True，表示第i个GT和第j个网格对应的anchor能匹配上
        """
        is_in_centers = center_deltas.min(dim=-1).values > 0.0

        """获得正样本的索引
        # 计算每个网格能与多少个GT进行匹配，>0表示对应的对应的anchor至少存在一个GT与之匹配
        # 返回值的维度为(8400, )
        """
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # 按照上述两种方法，只要有一种能和标签匹配上，就认为其是正样本, 维度为(8400, )"""
        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        """从第一种筛选方法得到的anchor中，筛选出能够通过第二种方法的anchor
        # 注意，参与&的两个矩阵，他们的列索引都是 is_in_boxes_anchor，且is_in_boxes_all在|前，也就说，是从第一种的方法的结果中进行筛选
        # 维度为(num_gt, len_first)，len_first是is_in_boxes_anchor中True的数量
        """
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        # 一个GT能和多个anchor进行匹配，但一个anchor只能和一个GT进行匹配，也就是说GT和anchor是一对多的关系
        # 这个函数先进行第二轮筛选，获得若干个anchor，然后求这些anchor与对应GT、GT的目标类别、与所匹配GT的IOU
        # 本函数还以传引用的方式对fg_mask进行了更新，更新后的fg_mask变成了第二轮筛选后得到的anchor在8400个anchor中的布尔索引
        :param cost: 第一轮筛选得到的anchor与GT的成本函数，维度为(num_gt, len_fg)，len_fg是8400个anchor经过第一轮筛选后得到的数量
        :param pair_wise_ious: GT和第一轮得到的anchor的交并比，维度为(num_gt, len_fg)
        :param gt_classes: 当前图片中，所有GT的类别索引，维度为(num_gt,)
        :param num_gt: 当前图片中GT的数量，纯数字
        :param fg_mask: 第一轮筛选得到的anchor在8400个anchor中的布尔索引，维度为(8400, )
        :return: num_fg: 当前图片中，所有GT的正样本总数，即能与任意一个GT匹配的anchor总数，一个纯数字
                 gt_matched_classes: 第二轮筛选后得到的anchor对应GT的类别索引，维度为(len_sg, )，len_sg是经过第二轮筛选后得到的anchor数量
                 pred_ious_this_matching: 第二轮筛选得到的anchor，与其对应的GT的iou，维度为(len_sg, )
                 matched_gt_inds:第二轮筛选得到的anchor能和哪些GT匹配，维度为(len_sg, )
        """

        """初始化匹配矩阵
        # 维度为(num_gt, len_fg)
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious  # (num_gt, len_fg)
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))  # 看10和len_fg哪个小

        # 对每个GT，寻找最大的10个（或len_fg个）IOU，维度为(num_gt, n_candidate_k)
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)

        """获得每个GT的k，即每个GT，该与多少个网格进行匹配
        # 上一步中，对各个GT求最大的k个IOU，这里topk_ious.sum(1)是将这k个最大的IOU进行求和操作，返回张量的维度为(num_gt, )
        # 可能某个GT，其最大的k个IOU相加都不不超过1，那么对这个GT，它的k就为1
        """
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()  # 转换为list

        """给每个真实框选取k个标签进行匹配
        # largest=False表示取最小
        # pos_idx是损失函数最小的k个预测框（anchor）对应的索引
        """
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx  # 释放内存

        anchor_matching_gt = matching_matrix.sum(0)  # 维度为(len_fg, )，表示每个anchor能和多少个GT进行匹配

        """
        # anchor_matching_gt > 1 的返回值是一个维度为(len_fg, )的布尔索引
        # 在len_fg个anchor中，如果存在某个anchor能和多个GT匹配，那么这个anchor对应的索引就是True
        # sum()用来求有多少个这样的特征点
        """
        if (anchor_matching_gt > 1).sum() > 0:
            """当某一个anchor指向多个GT的时候，选取cost最小的GT作为与其匹配的GT
            # cost[:, anchor_matching_gt > 1] 是将能与多个GT匹配的anchor取出，维度为(num_gt, match_mul)，
            # 某些anchor能与多个GT匹配，这样的anchor数量为match_mul，即match_mul是能与多个GT匹配的anchor的数量
            # torch.min dim=0表示对每列求最小值
            # cost_argmin每列最小值所对应的索引（GT的索引），维度为(match_mul, )
            # 若cost_argmin[2]为4，则表示在cost矩阵中，第2个anchor（match_mul中的第2个anchor）所在列中，与第4个GT的损失函数最小
            """
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)

            # 在matching_matrix中，先把这样的anchor所在列全部设为0，
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            # 再把每个这样的anchor列的最小值所对应的GT设为1
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        # 返回一个布尔索引，代表第一轮筛选得到的anchor是否能通过第二轮筛选，即是否为正样本，维度为(len_fg, )
        fg_mask_inboxes = matching_matrix.sum(0) > 0

        # 当前图片中，所有GT的正样本总数，即能与任意一个GT匹配的anchor总数，即len_sg
        num_fg = fg_mask_inboxes.sum().item()

        """对fg_mask进行更新
        # fg_mask本身代表8400个anchor中，通过第一轮筛选的anchor所对应的布尔索引
        # fg_mask[fg_mask.clone()]，布尔索引的布尔索引，即把所有为True的元素筛选出来，对这些元素进行重新赋值，
        # 赋值之后，fg_mask代表第二轮筛选后得到的anchor在8400个anchor中的索引
        # 至此，第二轮筛选结束，fg_mask的维度为(8400, )
        """
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        """获得第二轮筛选后得到的anchor，其所对应GT、GT的目标类别、与所匹配GT的IOU
        # 获得anchor对应GT的索引，维度为(len_sg, )
        # matching_matrix[:, fg_mask_inboxes]返回的是GT与第二轮筛选得到的anchor的匹配矩阵，维度为(num_fg, len_sg)
        # argmax(0)是求各列的最大值，因为各列只有一个值为1，其余都为0，由于每个anchor最多只能和一个GT匹配
        # 所以这里是求各个anchor能和哪些GT匹配，维度为(len_sg, )
        # 假设len_sg=20，即通过第二轮筛选后还剩20个anchor，
        # 若matched_gt_inds[5]的值为3，那么意思是第5个anchor（20中的第5个）匹配的GT的索引是3
        """
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        """根据GT的索引，获得特征点对应的GT的类别
        # gt_matched_classes维度为(len_sg, )
        # 若gt_matched_classes[5]的值为2，那么意思是第5个anchor（20中的第5个）匹配的GT，其类别索引为2
        """
        gt_matched_classes = gt_classes[matched_gt_inds]

        """
        # matching_matrix * pair_wise_ious的维度是(num_gt, len_fg)，表示经第二轮筛选后得到的anchor与GT的iou，
        # 每列最多只有一个元素有值，有可能一个都没有，所以sum(0)是将这些iou给取出来，变成一个维度为(fg_mask, )的张量，
        # [fg_mask_inboxes]是从中取出经过第二轮筛选后得到的anchor与对应的GT的iou
        # 最后得到的pred_ious_this_matching，其维度为(len_sg, )，表示第二轮筛选得到的anchor，与其对应的GT的iou
        # 若pred_ious_this_matching[5]的值为0.53，则表示第5个anchor（20中的第5个），与其匹配的GT的iou为0.53
        # 第5个anchor与哪一个GT匹配呢，这个要看 matched_gt_inds 才知道
        """
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
