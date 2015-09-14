const char* dgemm_desc = "My super awesome dgemm.";

#include <stdio.h>
#include <math.h>

#ifndef block_size
#define block_size ((int) 32)
#endif

#ifndef rect_length
#define rect_length ((int) 512)
#endif

// Blocked type
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
  const int n_rect = M/rect_length + (M%rect_length? 1 : 0);
  const int n_blocks = M / block_size + (M%block_size? 1 : 0);
  
  // Block loop
  for(int J=0; J<n_blocks; ++J) {
    for(int K=0; K<n_blocks; ++K) {
      for(int I=0; I<n_rect; ++I){
        //Inner block loop
          int j_end = ((J+1)*block_size < M? block_size : (M-(J*block_size)));
          for(int j=0;j< j_end; ++j) {
            int Cind = (I*rect_length)+(J*block_size+j)*M;
            int k_end = ((K+1)*block_size < M? block_size : (M-(K*block_size)));
            for(int k=0; k<k_end; ++k) {
              int Aind = (I*rect_length)+(K*block_size+k)*M;
              int Bind = (K*block_size)+k+(J*block_size+j)*M;
              int i_end = ((I+1)*rect_length < M? rect_length : (M-(I*rect_length)));
              for(int i=0; i< i_end; ++i) {
                C[Cind+i] += A[Aind+i]*B[Bind];
              }
            }
          }
      }
    }
  }
}
