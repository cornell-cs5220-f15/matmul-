const char* dgemm_desc = "My awesome dgemm.";

#include <nmmintrin.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B,
                 double* restrict C)//, const double restrict *C_original)
{
    // New kernal function for A stored in row-major.
    int i, j, k;
    for(j = 0; j < N; ++j){
      for(i = 0; i < M; ++i){
        double cij = C[i + j*lda];
        for (k = 0; k < K; ++k){
          cij += A[ i * BLOCK_SIZE + k ] * B[k + j * lda ];
        }
        C[i + j*lda] = cij;
      }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C, double *At,
              const int i, const int j, const int k)
{
    // Determine the size of each sub-block
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    basic_dgemm(lda, M, N, K, At, B + k + j*lda, C + i + j*lda);
                /*A + i + k*lda*/
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Preallocate a space for matrix A
    double* A_transposed = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // Assign blocks for kernals to perform fast computation.
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0); // # of blocks
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi){
      const int i = bi * BLOCK_SIZE;
      for (bk = 0; bk < n_blocks; ++bk){
        const int k = bk * BLOCK_SIZE;
        // Transpose A. This part needs to be rewritten for clarity and performance
        const int A_start = i + k*M;
        const int M_sub = (i+BLOCK_SIZE > M? M-i : BLOCK_SIZE);
        const int K = (k+BLOCK_SIZE > M? M-k : BLOCK_SIZE);
        int it, kt;
        // printf("A Start is %d\n", A_start);
        for (it = 0; it < M_sub; ++it){
          for (kt = 0; kt < K; ++kt){
            A_transposed[it*BLOCK_SIZE + kt] = A[A_start + it + kt*M];
          }
        }

        for (bj = 0; bj < n_blocks; ++bj){
          const int j = bj * BLOCK_SIZE;
          do_block(M, A, B, C, A_transposed, i, j, k);
        }
      }
    }
    free(A_transposed);
}
