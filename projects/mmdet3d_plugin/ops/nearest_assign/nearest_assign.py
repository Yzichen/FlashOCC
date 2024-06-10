# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import nearest_assign_ext

__all__ = ['nearest_assign']


class QuickNearestAssignCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                occ_pred, 
                l2s_key,
                occind2detind,
                inst_cls,
                inst_xyz,
                inst_id_list,
                ):

        occ_pred = occ_pred.contiguous().int()
        l2s_key = l2s_key.contiguous().int()
        occind2detind = occind2detind.contiguous().int()
        inst_cls = inst_cls.contiguous().int()
        inst_xyz = inst_xyz.contiguous().int()
        inst_id_list = inst_id_list.contiguous().int()
        inst_pred = occ_pred.new_zeros(occ_pred.shape)

        nearest_assign_ext.nearest_assign_forward(
            occ_pred, 
            l2s_key,
            occind2detind,
            inst_cls,
            inst_xyz,
            inst_id_list,
            inst_pred
        )

        return inst_pred


def nearest_assign(occ_pred, 
                l2s_key,
                occind2detind,
                inst_cls,
                inst_xyz,
                inst_id_list):
    inst_pred = QuickNearestAssignCuda.apply(occ_pred, 
                l2s_key,
                occind2detind,
                inst_cls,
                inst_xyz,
                inst_id_list
                )      # (B, Dz, Dy, Dx, C)
    return inst_pred

def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float().cuda()
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
    feat = torch.ones(
        size=[1, 1, 2, 2, 2], dtype=torch.float,
        device='cuda').requires_grad_()
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().cuda()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().cuda()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().cuda()

    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           (1, 1, 2, 2, 2), interval_starts, interval_lengths)
    loss = torch.sum(bev_feat)
    loss.backward()
    assert loss == 4.4
    grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.cuda().view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
    grad_feat = torch.from_numpy(grad_feat).float().cuda().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
