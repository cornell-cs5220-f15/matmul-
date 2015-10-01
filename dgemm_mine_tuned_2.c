#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif
#ifndef BYTE_ALIGNMENT
#define BYTE_ALIGNMENT ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void basic_dgemm_copied(const int M, const int N, const int K,
                 const double *restrict A, const double *restrict B, double *restrict C)
{
    int i, j, k;
    for (k = 0; k < K; ++k) {
        for (j = 0; j < N; ++j) {
            for (i = 0; i < M; ++i) {
                C[j*M+i] = A[k*M+i] * B[j*K+k];
            }

        }
    }
}

void do_block(const int lda,
              const double *restrict AA, const double *restrict BB, double *restrict CC,
              const double *restrict A, const double *restrict B, double *restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    for (int kk = 0; kk < K; kk++) {
        memcpy((void *) (AA + (kk * M)), (const void *) (A + i + (k + kk)*lda), M * sizeof(double));
    }
    for (int jj = 0; jj < N; jj++) {
        memcpy((void *) (BB + (jj * K)), (const void *) (B + k + (j + jj)*lda), K * sizeof(double));
        memcpy((void *) (CC + (jj * M)), (const void *) (C + i + (j + jj)*lda), M * sizeof(double));
    }

    basic_dgemm_copied(M, N, K, AA, BB, CC);

    for (int jj = 0; jj < N; jj++) {
        memcpy((void *) (C + i + (j + jj)*lda), (const void *) (CC + (jj * M)), M * sizeof(double));
    }
}

void square_dgemm(const int MM, const double *restrict A, const double *restrict B, double *restrict C)
{
    const double *AA = (double *) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGNMENT);
    const double *BB = (double *) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGNMENT);
    double *CC = (double *) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGNMENT);

    const int n_blocks = MM / BLOCK_SIZE + (MM%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bk = 0; bk < n_blocks; ++bk) {
        const int k = bk * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                if (i+BLOCK_SIZE > MM || j+BLOCK_SIZE > MM || k+BLOCK_SIZE > MM) {
                  const int M = (i+BLOCK_SIZE > MM? MM-i : BLOCK_SIZE);
                  const int N = (j+BLOCK_SIZE > MM? MM-j : BLOCK_SIZE);
                  const int K = (k+BLOCK_SIZE > MM? MM-k : BLOCK_SIZE);
                  const double *AAA = (double *) _mm_malloc(M * K * sizeof(double), BYTE_ALIGNMENT);
                  const double *BBB = (double *) _mm_malloc(K * N * sizeof(double), BYTE_ALIGNMENT);
                  double *CCC = (double *) _mm_malloc(M * N * sizeof(double), BYTE_ALIGNMENT);
                  do_block(MM, AAA, BBB, CCC, A, B, C, i, j, k);
                  _mm_free((void *) AAA);
                  _mm_free((void *) BBB);
                  _mm_free((void *) CCC);
                } else {
                  do_block(MM, AA, BB, CC, A, B, C, i, j, k);
                }
            }
        }
    }

    _mm_free((void *) AA);
    _mm_free((void *) BB);
    _mm_free((void *) CC);
}
