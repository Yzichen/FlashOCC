// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>

__global__ void nearest_assign_kernel(
                                  const int* l2s_key,
                                  int l2s_size,
                                  const int* occind2detind,
                                  const int *__restrict__ occ_pred,
                                  const int *__restrict__ inst_xyz,
                                  const int *__restrict__ inst_cls,
                                  const int *__restrict__ inst_id_list,
                                  int inst_size,
                                  int* __restrict__ inst_pred) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 该pillar的cur_c特征对应的索引.
  // while (idx < 200*200*16)
  if (true)
  {

    int occ_pred_label = occ_pred[idx];
    int dist_min = 100000000;
    // printf("idx: %d \n", idx);
    // int tgt_inst_id = -1;
    for (int index = 0; index < l2s_size; index ++)
    {
      // printf("occ_pred_label: %d == l2s_key[%d]:, %d \n", occ_pred_label, index, l2s_key[index]);
      // if (abs(occ_pred_label - l2s_key[index]) < 1)
      if (occ_pred_label == l2s_key[index])
      {
        // printf("occ_pred_label: %d == l2s_key[%d]:, %d \n", occ_pred_label, index, l2s_key[index]);
        int x = idx/(200*16);
        int y = (idx - x*200*16)/16;
        int z = idx - x*200*16 - y*16;
        int inst_ind = 0;
        for (inst_ind = 0; inst_ind < inst_size; inst_ind ++)
        {
          // printf("inst_ind[%d]:%d, occind2detind[%d]:%d \n", inst_ind, inst_cls[inst_ind], occ_pred_label, occind2detind[occ_pred_label]);
          // printf("inst_cls[%d]:%d == occind2detind[%d]:%d \n", inst_ind, inst_cls[inst_ind], occ_pred_label, occind2detind[occ_pred_label]);
          if (inst_cls[inst_ind] == occind2detind[occ_pred_label])
          {
            int dx = x - inst_xyz[inst_ind*3+0];
            int dy = y - inst_xyz[inst_ind*3+1];
            int dz = z - inst_xyz[inst_ind*3+2];
            int dist = dx*dx + dy*dy + dz*dz;
            // // // printf("idx:%d, inst_pred[%d,%d,%d]=%d, inst_ind/inst_size:%d/%d , dist/dist_min=%d/%d \n", idx, x,y,z, inst_pred[idx], inst_ind, inst_size, dist, dist_min);
            if (dist < dist_min){
              dist_min = dist;
              inst_pred[idx] = inst_id_list[inst_ind];
              // printf("idx:%d, inst_pred[%d,%d,%d]=%d, inst_ind/inst_size:%d/%d , dist/dist_min=%d/%d, occind2detind[%d]=%d \n", idx, x,y,z, inst_pred[idx], inst_ind, inst_size, dist, dist_min, occ_pred_label, occind2detind[occ_pred_label]);
              // // // printf("inst_pred[%d,%d,%d]=%d, inst_ind/inst_size:%d/%d , dist_min=%d \n", x,y,z, inst_pred[idx], inst_ind, inst_size, dist_min);
            }
          }
        }
        return;
      }
    }
    inst_pred[idx] = occ_pred[idx];

    // idx += blockDim.x * gridDim.x;
  }

}

void nearest_assign(
              const int* l2s_key,
              int l2s_size,
              const int *__restrict__ occind2detind,
              int inst_size,
              const int *__restrict__ occ_pred,
              const int *__restrict__ inst_xyz,
              const int *__restrict__ inst_cls,
              const int *__restrict__ inst_id_list,
              int* __restrict__ inst_pred) {
  // nearest_assign_kernel<<<128, 256>>>(
  nearest_assign_kernel<<<(int)ceil(((double)200 * 200 * 16 / 256)), 256>>>(
    l2s_key, l2s_size, occind2detind, 
    occ_pred, inst_xyz, inst_cls, 
    inst_id_list, inst_size, inst_pred
  );
}


