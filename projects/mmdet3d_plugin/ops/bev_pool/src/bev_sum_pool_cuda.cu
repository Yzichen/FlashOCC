#include <stdio.h>
#include <stdlib.h>
#include "bev_sum_pool.h"

/*
  Function: pillar pooling
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    geom_feats       : input features, FloatTensor[n, c]
    geom_coords      : input coordinates, IntTensor[n, 4]  4: (x_id, y_id, z_id, batch_id)
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
__global__ void bev_sum_pool_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ geom_feats,
                                  const int *__restrict__ geom_coords,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_geom_coords = geom_coords + interval_start * 4;    // 当前负责计算的pillar的坐标 4: (x_id, y_id, z_id, batch_id)
  const float* cur_geom_feats = geom_feats + interval_start * c + cur_c;
  float* cur_out = out + cur_geom_coords[3] * d * h * w * c +
    cur_geom_coords[2] * h * w * c + cur_geom_coords[1] * w * c +
    cur_geom_coords[0] * c + cur_c;
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_geom_feats[i * c];
  }
  *cur_out = psum;
}


/*
  Function: pillar pooling backward
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    geom_coords       : input coordinates, IntTensor[N, 4]  4: (x_id, y_id, z_id, batch_id)
    interval_lengths : how many points in each pooled point, IntTensor[n_intervals]
    interval_starts  : starting position for pooled point, IntTensor[n_intervals]
    x_grad           : gradient of the image fmap, FloatTensor
*/
__global__ void bev_sum_pool_grad_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ out_grad,
                                  const int *__restrict__ geom_coords,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ x_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  // 当前负责计算的pillar的坐标 4: (x_id, y_id, z_id, batch_id)
  // 该pillar中所有点的梯度 与 该pillar特征的梯度相同.
  const int* cur_geom_coords = geom_coords + interval_start * 4;
  float* cur_x_grad = x_grad + interval_start * c + cur_c;
  
  const float* cur_out_grad = out_grad + cur_geom_coords[3] * d * h * w * c +
    cur_geom_coords[2] * h * w * c + cur_geom_coords[1] * w * c +
    cur_geom_coords[0] * c + cur_c;
  for(int i = 0; i < interval_length; i++){
    cur_x_grad[i * c] = *cur_out_grad;
  }
  
}

void bev_sum_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* geom_feats,
  const int* geom_coords, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_sum_pool_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, geom_feats, geom_coords, interval_starts, interval_lengths, out
  );
}

void bev_sum_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_coords, const int* interval_starts, const int* interval_lengths, float* x_grad) {
  bev_sum_pool_grad_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, out_grad, geom_coords, interval_starts, interval_lengths, x_grad
  );
}
