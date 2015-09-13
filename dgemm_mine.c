const char* dgemm_desc = "My super awesome dgemm.";

#include <stdio.h>
#include <math.h>

#ifndef block_size
#define block_size ((int) 8)
#endif

#ifndef rect_length
#define rect_length ((int) 256)
#endif

// Blocked type
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
  const int n_rect = M/rect_length + (M%rect_length? 1 : 0);
  const int n_blocks = M / block_size + (M%block_size? 1 : 0);
  
  // Block loop
  for(int J=0; J<n_blocks; ++J) {
    for(int K=0; K<n_blocks; ++K) {
      for(int I=0; I<n_rect; ++I) {
        //Inner block loop
          int j_end = ((J+1)*block_size < M? (J+1)*block_size : M);
          for(int j=J*block_size;j< j_end; ++j) {
            int k_end = ((K+1)*block_size < M? (K+1)*block_size : M);
            for(int k=K*block_size; k<k_end; ++k) {
              int i_end = ((I+1)*rect_length < M? (I+1)*rect_length : M);
              for(int i=I*rect_length; i< i_end; ++i) {
                C[i+j*M] += A[i+k*M]*B[k+j*M];
              }
            }
          }
      }
    }
  }
}

//// Rectangular column sub-section type
//void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
//{
//  const int n_block = M / block_size + (M%block_size? 1 : 0);
//  
//  for(int J=0; J<n_block; ++J) {
//    for(int K=0; K<n_block; ++K) {
//      int j_end = ((J+1)*block_size < M? (J+1)*block_size : M);
//      for(int j=J*block_size; j<j_end; ++j) {
//        int k_end = ((K+1)*block_size < M? (K+1)*block_size : M);
//        for(int k=K*block_size; k<k_end; ++k) {
//          for(int i=0; i<M; ++i) {
//          C[i+j*M] += A[i+k*M]*B[k+j*M];
//          }
//        }
//      }
//    }
//  }
//  
//}
