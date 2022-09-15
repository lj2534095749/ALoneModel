#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """求GT与预测框（anchor）的交并比

    :param bboxes_a: GT，维度为(num_gt, 4)
    :param bboxes_b: 预测框，维度为(len_fg, 4)，len_fg是经过第一轮筛选后得到的anchor数量
    :param xyxy: GT和预测框，是否为边框上下角点的坐标
    :return: iou, GT和预测框的交并比，维度为(num_gt, len_fg)，例如iou[i, j]表示第i个GT和第j个预测框的交并比
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        """
        # tl 重合区域的左边缘和上边缘，维度为(num_gt, len_fg, 2)
        # br 重合区域的下边缘和右边缘，维度为(num_gt, len_fg, 2)
        # bboxes_a[:, None, :2]等价于先获得切片bboxes_a[:, :2]，再使用unsqueeze(1)，得到的张量维度为(num_gt, 1, 2)
        # bboxes_b[:, :2]的维度为(len_fg, 2)，而bboxes_a[:, None, :2]的维度是(num_gt, 1, 2)，两者可以进行广播
        """
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  # 方框a的面积，维度为(num_gt, )
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)  # 方框b的面积，维度为(len_fg, )
    else:
        """
        # tl 重合区域的左边缘和上边缘，维度为(num_gt, len_fg, 2)
        # br 重合区域的下边缘和右边缘，维度为(num_gt, len_fg, 2)
        """
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)  # 方框a的面积，维度为(num_gt, )
        area_b = torch.prod(bboxes_b[:, 2:], 1)  # 方框b的面积，维度为(len_fg, )

    """获得左上小于右下的索引, 维度为(num_gt, len_fg)
    # (tl < br)得到布尔索引，(num_gt, len_fg, 2)
    # type(tl.type())将其转化为数值，(num_gt, len_fg, 2)
    # prod(dim=2)表示将第二个维度的元素相乘，(num_gt, len_fg)
    # 如果相乘之后还是1，那么说明“左<右”和“上<下”同时满足，即GT和预测框存在交集
    """
    en = (tl < br).type(tl.type()).prod(dim=2)

    """求交集(左上小于右下)面积
    先求交集面积，然后通过 * en，对存在交集的面积进行筛选，area_i的维度为(num_gt, len_fg)
    """
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)
