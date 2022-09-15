import numpy as np
import torch
import torch.nn as nn
import importlib
import sys
import os
from torch.utils.data.dataset import Dataset

if __name__ == "__main__":
    pass


    # print(os.path.split(os.path.realpath(__file__))[1].split(".")[0])

    # exp_file = "temp.py"
    # try:
    #     sys.path.append(os.path.dirname(exp_file))
    #     current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    #     exp = current_exp.test()
    #     print(exp.a)
    # except Exception:
    #     raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))

    # print(torch.cuda.is_available())  # cuda是否可用
    # print(torch.cuda.device_count())  # gpu数量
    # print(torch.cuda.current_device())  # 当前设备索引, 从0开始
    # print(torch.cuda.get_device_name(0))  # 返回gpu名字

    # @torch.no_grad()
    # def init_yolo(M):
    #     for m in M.modules():
    #         print("-----------------------------")
    #         print(m)
    #         print(isinstance(m, nn.BatchNorm2d))
    #
    #
    # net = nn.Sequential(nn.BatchNorm2d(8))
    # # print(net)
    # # print('isinstance torch.nn.Module', isinstance(net, torch.nn.Module))
    # # print(' ')
    # net.apply(init_yolo)

    # a = torch.tensor([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    # ])
    # print(a)
    # print(a[..., :2])
    # print(a[::2, 1::2])

    # print(round(1.5))

    # matching_matrix = torch.tensor([
    #     [0, 1, 0],
    #     [1, 1, 0],
    #     [0, 0, 0]
    # ], dtype=float)
    # anchor_matching_gt = matching_matrix.sum(0)
    # print(anchor_matching_gt)
    # cost = torch.ones(3, 3)
    # cost[1][1] = 2
    # print(cost)
    # print(anchor_matching_gt > 1)
    # print(cost[:, anchor_matching_gt > 1])
    # print(torch.min(cost[:, anchor_matching_gt > 1], dim=0))

    # topk_ious = torch.ones(2, 3)
    # print(topk_ious)
    # dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    # print(dynamic_ks, dynamic_ks.shape)
    # dynamic_ks = dynamic_ks.tolist()
    # print(dynamic_ks)
    # ious_in_boxes_matrix = torch.ones(2, 3)
    # topk_ious, _ = torch.topk(ious_in_boxes_matrix, 2, dim=1)
    # print(topk_ious, _)

    # bboxes_a = torch.ones(2, 2)
    # bboxes_b = torch.ones(4, 2) * 2
    # tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
    # print(bboxes_a)
    # print(bboxes_b)
    # print(tl.shape)
    # print(tl)
    # bboxes_aa = torch.ones(2, 2) * 2
    # bboxes_bb = torch.ones(4, 2) * 4
    # br = torch.max(bboxes_aa[:, None, :2], bboxes_bb[:, :2])
    # br[0][0][0] = 1
    # print(bboxes_aa)
    # print(bboxes_bb)
    # print(br.shape)
    # print(br)
    # en = (tl < br).type(tl.type()).prod(dim=2)
    # print(en.shape, en)
    # area_i = torch.prod(br - tl, 2)
    # print(area_i, area_i.shape)
    # print(area_i*en)

    # yv, xv = torch.meshgrid([torch.arange(4), torch.arange(4)])
    # print(yv)
    # print(xv)

    # x_shift = []
    # x_shift.append(torch.ones(1, 10))
    # x_shift.append(torch.zeros(1, 5))
    # x_shift.append(torch.ones(1, 2))
    # print(x_shift)
    # print(torch.cat(x_shift, 1))

    # bbox_deltas = torch.ones(2, 10, 4)
    # is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    # is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    # print(bbox_deltas.shape)
    # print(is_in_boxes.shape)
    # print(is_in_boxes_all.shape)
    # print(bbox_deltas)
    # print(is_in_boxes)
    # print(is_in_boxes_all)
    # print("----------")
    # is_in_boxes_all[0] = False
    # print(is_in_boxes_all)
    # bboxes_preds_per_image = torch.ones(10, 4)
    # bboxes_preds_per_image = bboxes_preds_per_image[is_in_boxes_all]
    # print(bboxes_preds_per_image.shape)
    # print(bboxes_preds_per_image)

    # [:] demo
    # grids = [torch.zeros(1)] * 3
    # print(grids)
    # print(grids[0].shape[2:4])
    # a = [0, 1, 2, 3]
    # print(a[2:4])
    # print(a[2:])
    # print(a[:2])

    # grid demo
    # yv, xv = torch.meshgrid([torch.arange(5), torch.arange(5)])
    # print(yv)
    # print(xv)
    # print("-------------------")
    # grid = torch.stack((xv, yv), 2).view(1, 1, 5, 5, 2)
    # print(torch.stack((xv, yv), 2).shape)
    # print(grid.shape)
    # print(grid.view(1, -1, 2).shape)
    # print(grid.view(1, -1, 2).shape)
    # grid = grid.view(1, -1, 2)
    # print(grid[..., :2].shape)
    # # print(grid[..., :2] * 6)
    # print(torch.exp(grid[..., 2:4]) * 8)
    # print("-------------------")
    # x_shifts = []
    # y_shifts = []
    # expanded_strides = []
    # x_shifts.append(grid[:, :, 0])  # 获得第k个检测头每个网格点在x方向上的偏移
    # y_shifts.append(grid[:, :, 1])  # 获得第k个检测头每个网格点在y方向上的偏移
    # expanded_strides.append(  # 表示每个网格的步长，维度和x_shifts相同
    #     torch.zeros(1, grid.shape[1]).fill_(8)
    # )
    # print(expanded_strides[0].shape)
    # print(x_shifts[0].shape)
    # print(y_shifts[0].shape)
    # print("-------------------")
    # grids = []
    # grids.append(grid)
    # grids.append(grid)
    # grids.append(grid)
    # print(grid.shape)
    # print(torch.cat(grids, 1).shape)

    # enumerate zip demo
    # seq1 = ['foo', 'bar', 'baz']
    # seq2 = ['one', 'two', 'three']
    # for i, (a, b) in enumerate(
    #         zip(seq1, seq2)
    # ):
    #     print('{}:({},{})'.format(i, a, b))

    # upsample demo
    # input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    # print(input.shape)
    # m = nn.Upsample(scale_factor=2, mode='nearest')
    # print(m(input).shape)

    # maxpool demo
    # m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # input = torch.randn(20, 16, 50, 32)
    # output = m(input)
    # print(output.shape)

    # for _ in range(3):
    #     print(_)

    # * demo
    # a = torch.tensor(np.array(
    #     [[[1, 1],
    #       [1, 1],
    #       [1, 1]],
    #      [[1, 1],
    #       [1, 1],
    #       [1, 1]]]
    # ), dtype=torch.float)
    # b = torch.tensor(np.array(
    #     [[[1, 2]],
    #      [[3, 4]]]
    # ), dtype=torch.float)
    # print(b.size())
    # print(a * b)
    # output = m(input)
    # print(output)

    # reshape view demo
    # b = torch.tensor(
    #     [[0, 1, 2],
    #      [3, 9, 5]])
    # stride = b.stride()
    # size = b.size()
    # print(stride)
    # print(size)
    # print(stride[1])

    # Linear demo
    # m = nn.Linear(2, 3, bias=False)
    # input = torch.tensor(np.array(
    #     [[1, 2],
    #      [3, 4],
    #      [5, 6]]
    # ), dtype=torch.float)
    # print(input)
    # output = m(input)
    # print(output)
