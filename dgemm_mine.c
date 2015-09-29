const char* dgemm_desc = "My super awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>


#ifndef block_size
#define block_size ((int) 64)
#endif

#ifndef rect_length
#define rect_length ((int) 384)
#endif

// Blocked type
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
  int I,J,K,i,j,k,i_end,j_end,k_end,Aind,Bind,Cind;
  int B_block_ind, A_block_ind;
  double BRHS;
  const int n_rect = M/rect_length + (M%rect_length? 1 : 0);
  const int n_blocks = M / block_size + (M%block_size? 1 : 0);

  double* A_block = (double*) mm_malloc(rect_length * block_size * sizeof(double),64);
  double* B_block = (double*) mm_malloc(block_size * block_size * sizeof(double),64);
  
  // Block loop
  for(J=0; J<n_blocks; ++J) {
    for(K=0; K<n_blocks; ++K) {
      for(I=0; I<n_rect; ++I){
        j_end = ((J+1)*block_size < M? block_size : (M-(J*block_size)));
        // Perform Copy
        for(j=0;j< j_end; ++j) {
          k_end = ((K+1)*block_size < M? block_size : (M-(K*block_size)));
          B_block_ind = j*block_size;
          Bind = (K*block_size)+(J*block_size+j)*M;
          for(k=0; k<k_end; ++k) {
            Aind = (I*rect_length)+(K*block_size+k)*M;
            A_block_ind = k*rect_length;
            B_block[B_block_ind] = 0.0;
            B_block[B_block_ind+k] = B[Bind+k];
            i_end = ((I+1)*rect_length < M? rect_length : (M-(I*rect_length)));
            for(i=0; i< i_end; ++i) {
              A_block[A_block_ind+i] = A[Aind+i];
              }
            }
          }
        //Perform Multiply
          for(j=0;j< j_end; ++j) {
            Cind = (I*rect_length)+(J*block_size+j)*M;
            B_block_ind = j*block_size;
            for(k=0; k<k_end; ++k) {
              A_block_ind = k*rect_length;
              for(i=0; i< i_end; i=i+8) {
                C[Cind+i] += A[A_block_ind+i]*B_block[B_block_ind+k];
                C[Cind+i+1] += A[A_block_ind+i+1]*B_block[B_block_ind+k+1];
                C[Cind+i+2] += A[A_block_ind+i+2]*B_block[B_block_ind+k+2];
                C[Cind+i+3] += A[A_block_ind+i+3]*B_block[B_block_ind+k+3];
                C[Cind+i+4] += A[A_block_ind+i+4]*B_block[B_block_ind+k+4];
                C[Cind+i+5] += A[A_block_ind+i+5]*B_block[B_block_ind+k+5];
                C[Cind+i+6] += A[A_block_ind+i+6]*B_block[B_block_ind+k+6];
                C[Cind+i+7] += A[A_block_ind+i+7]*B_block[B_block_ind+k+7];
              }
            }
          }
      }
    }
  }
  
  mm_free(A_block);
  mm_free(B_block);
}
