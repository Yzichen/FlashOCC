#ifndef _BEV_MAX_POOL_H
#define _BEV_MAX_POOL_H

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

at::Tensor bev_max_pool_forward(
  const at::Tensor _geom_feats,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
);

at::Tensor bev_max_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
);


// CUDA function declarations
void bev_max_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_max_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);


#endif