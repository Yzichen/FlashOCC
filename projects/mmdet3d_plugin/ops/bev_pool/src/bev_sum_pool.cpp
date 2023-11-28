#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "bev_sum_pool.h"

/*
  Function: pillar pooling (forward, cuda)
  Args:
    geom_feats         : input features, FloatTensor[N, C]
    _geom_coords       : input coordinates, IntTensor[N, 4]  4: (x_id, y_id, z_id, batch_id)
    interval_lengths : how many points in each pooled point, IntTensor[N_pillar, ]
    interval_starts  : starting position for pooled point, IntTensor [N_pillar, ]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_sum_pool_forward(
  const at::Tensor _geom_feats,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _geom_feats.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_geom_feats));
  const float* geom_feats = _geom_feats.data_ptr<float>();
  const int* geom_coords = _geom_coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options =
      torch::TensorOptions().dtype(_geom_feats.dtype()).device(_geom_feats.device());
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);     // (B, D=Dz, H=Dy, W=Dx, C)
  float* out = _out.data_ptr<float>();
  bev_sum_pool(
    b, d, h, w, n, c, n_intervals, geom_feats,
    geom_coords, interval_starts, interval_lengths, out
  );
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[B, D, H, W, C]
    geom_coords       : input coordinates, IntTensor[N, 4]
    interval_lengths : how many points in each pooled point, IntTensor[N_pillar, ]
    interval_starts  : starting position for pooled point, IntTensor [N_pillar, ]
  Return:
    x_grad           : output features, FloatTensor[N, C]
*/
at::Tensor bev_sum_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_coords.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_coords = _geom_coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);   // (N, C)
  float* x_grad = _x_grad.data_ptr<float>();
  
  bev_sum_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad,
    geom_coords, interval_starts, interval_lengths, x_grad
  );
  
  return _x_grad;
}