# Copyright (c) OpenMMLab. All rights reserved.
import numba
import numpy as np
import torch
from mmcv.ops import nms, nms_rotated


# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None,
            xyxyr2xywhr=True):
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    if xyxyr2xywhr:
        boxes = torch.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
            dim=-1)

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


# This function duplicates functionality of mmcv.ops.iou_3d.nms_normal_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms.
# Nms api will be unified in mmdetection3d one day.
def nms_normal_bev(boxes, scores, thresh):
    """Normal NMS function GPU implementation (for BEV boxes). The overlap of
    two boxes for IoU calculation is defined as the exact overlapping area of
    the two boxes WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'
    return nms(boxes[:, :-1], scores, thresh)[1]
