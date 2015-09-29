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
#define rect_length ((int) 256)
#endif

// Blocked type
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
  int I,J,K,i,j,k,i_end,j_end,k_end,Aind,Bind,Cind;
  int C_block_ind, B_block_ind, A_block_ind;
  double BRHS;
  const int n_rect = M/rect_length + (M%rect_length? 1 : 0);
  const int n_blocks = M / block_size + (M%block_size? 1 : 0);
    
  double* A_block = (double*) _mm_malloc(rect_length * block_size * sizeof(double),32);
  double* B_block = (double*) _mm_malloc(block_size * block_size * sizeof(double),32);
  double* C_block = (double*) _mm_malloc(rect_length*block_size*sizeof(double),32);
  
  __assume_aligned(A_block,32);
  __assume_aligned(B_block,32);
  __assume_aligned(C_block,32);
  __assume(Aind%4==0);
  __assume(Bind%4==0);
  __assume(Cind%4==0);

  // Block loop
  for(J=0; J<n_blocks; ++J) {
    for(K=0; K<n_blocks; ++K) {
      for(I=0; I<n_rect; ++I){
        j_end = ((J+1)*block_size < M? block_size : (M-(J*block_size)));
        k_end = ((K+1)*block_size < M? block_size : (M-(K*block_size)));
        i_end = ((I+1)*rect_length < M? rect_length : (M-(I*rect_length)));
        // Perform Copy
        for(j=0;j< j_end; ++j) {
          C_block_ind = j*rect_length;
          B_block_ind = j*block_size;
          Cind = (I*rect_length)+(J*block_size+j)*M;
          Bind = (K*block_size)+(J*block_size+j)*M;
         
        //  __assume_aligned(B_block,64);
          for(k=0; k<k_end; ++k) {
            B_block[B_block_ind+k] = B[Bind+k];
            Aind = (I*rect_length)+(K*block_size+k)*M;
            A_block_ind = k*rect_length;

         //  __assume_aligned(A_block,64);
         //  __assume_aligned(C_block,64);
            for(i=0; i< i_end; ++i) {
              A_block[A_block_ind+i] = A[Aind+i];
              C_block[C_block_ind+i] = C[Cind+i];
              }
            }
          }
        //Perform Multiply
          for(j=0;j< j_end; ++j) {
            C_block_ind = j*rect_length;
            for(k=0; k<k_end; ++k) {
              B_block_ind = j*block_size;
              A_block_ind = k*rect_length;

           //   __assume_aligned(C_block,64);
            //  __assume_aligned(B_block,64);
            //  __assume_aligned(A_block,64);
              for(i=0; i< i_end; ++i) {
                C_block[C_block_ind+i] += A_block[A_block_ind+i]*B_block[B_block_ind+k];
//                C[Cind+i+1] += A_block[A_block_ind+1]*B_block[B_block_ind+1];
//                C[Cind+i+2] += A_block[A_block_ind+2]*B_block[B_block_ind+2];
//                C[Cind+i+3] += A_block[A_block_ind+3]*B_block[B_block_ind+3];
//                C[Cind+i+4] += A_block[A_block_ind+4]*B_block[B_block_ind+4];
//                C[Cind+i+5] += A_block[A_block_ind+5]*B_block[B_block_ind+5];
//                C[Cind+i+6] += A_block[A_block_ind+6]*B_block[B_block_ind+6];
//                C[Cind+i+7] += A_block[A_block_ind+7]*B_block[B_block_ind+7];
              }
            }
          }
          for(j=0;j<j_end;++j) {
            Cind = (I*rect_length)+(J*block_size+j)*M;
            C_block_ind = j*rect_length;
            
           // __assume_aligned(C_block,64);
            for(i=0;i<i_end;++i){
	      C[Cind+i] = C_block[C_block_ind+i];
            }
         }
      }
    }
  }
  _mm_free(A_block);
  _mm_free(B_block);
  _mm_free(C_block);
}
