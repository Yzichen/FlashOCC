import torch

from . import bev_pool_ext


class QuickBevPoolingCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coords, ranks, B, D, H, W, pooling_method):
        """
        Args:
            ctx:
            feats: (N, C)
            coords: (N, 4)   4: (x_id, y_id, z_id, batch_id)
            ranks:  (N, )  eg: (0, 0, 1, 1, 1, 2, 2)
            B:
            D:
            H:
            W:
        Returns:
            out: (B, D, H, W, C)
        """
        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)    # (N, )
        kept[1:] = ranks[1:] != ranks[:-1]      # 边界点=1, 其余为0（pillar id发生变化）    eg:(1, 0, 1, 0, 0, 1, 0)
        interval_starts = torch.where(kept)[0].int()    # 该pillar的起始位置  (N_pillar, )    eg: (0, 2, 5)
        interval_lengths = torch.zeros_like(interval_starts)    # pillar包含points的数量  (N_pillar, )  eg: (0, 0, 0)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]   # eg: (0, 2, 5)
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]     # eg: (0, 3, 2)
        coords = coords.int()

        if pooling_method == 'sum':
            out = bev_pool_ext.bev_sum_pool_forward(
                feats,      # (N, C)
                coords,     # (N, 4)   4: (x_id, y_id, z_id, batch_id)
                interval_lengths,   # (N_pillar, )
                interval_starts,    # (N_pillar, )
                B,
                D,
                H,
                W,
            )
        elif pooling_method == 'max':
            out = bev_pool_ext.bev_max_pool_forward(
                feats,      # (N, C)
                coords,     # (N, 4)   4: (x_id, y_id, z_id, batch_id)
                interval_lengths,   # (N_pillar, )
                interval_starts,    # (N_pillar, )
                B,
                D,
                H,
                W,
            )

        ctx.save_for_backward(interval_starts, interval_lengths, coords)
        ctx.saved_shapes = B, D, H, W
        ctx.pooling_method = pooling_method
        return out

    @staticmethod
    def backward(ctx, out_grad):
        """
        Args:
            ctx:
            out_grad: (B, D, H, W, C)

        Returns:
            x_grad: (N, C)
        """
        # (N_pillar, ),  (N_pillar, ),  (N, 4)   4: (x_id, y_id, z_id, batch_id)
        interval_starts, interval_lengths, geom_coords = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes
        pooling_method = ctx.pooling_method

        out_grad = out_grad.contiguous()
        if pooling_method == 'sum':
            x_grad = bev_pool_ext.bev_sum_pool_backward(
                out_grad,               # (B, D, H, W, C)
                geom_coords,            # (N, 4)   4: (x_id, y_id, z_id, batch_id)
                interval_lengths,       # (N_pillar, )
                interval_starts,        # (N_pillar, )
                B,
                D,
                H,
                W,
            )   # (N, C)
        elif pooling_method == 'max':
            x_grad = bev_pool_ext.bev_max_pool_backward(
                out_grad,               # (B, D, H, W, C)
                geom_coords,            # (N, 4)   4: (x_id, y_id, z_id, batch_id)
                interval_lengths,       # (N_pillar, )
                interval_starts,        # (N_pillar, )
                B,
                D,
                H,
                W,
            )   # (N, C)

        return x_grad, None, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W, pooling_method='sum'):
    """
    Args:
        feats: (N, C)
        coords: (N, 4)  4: (x_id, y_id, z_id, batch_id)
        B:
        D:  Dz
        H:  Dy
        W:  Dx
    Returns:
        bev_features: (B, C, D, H, W)
    """
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (H * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )       # (N, )
    indices = ranks.argsort()   # (N, )
    # (N, C), (N, 4), (N, )
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickBevPoolingCuda.apply(feats, coords, ranks, B, D, H, W, pooling_method)     # (B, D, H, W, C)
    x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, C, D, H, W)
    return x
