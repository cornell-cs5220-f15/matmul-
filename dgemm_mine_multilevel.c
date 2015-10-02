#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE_1
#define BLOCK_SIZE_1 ((int) 104)
#endif
#ifndef BLOCK_SIZE_2
#define BLOCK_SIZE_2 ((int) 809)
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
                C[j*M+i] += A[k*M+i] * B[j*K+k];
            }

        }
    }
}

void square_dgemm_varblock(const int bs,
  const int MM, const double *restrict A, const double *restrict B, double *restrict C);

void do_block(const int bs,
              const int lda,
              const double *restrict AA, const double *restrict BB, double *restrict CC,
              const double *restrict A, const double *restrict B, double *restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+bs > lda? lda-i : bs);
    const int N = (j+bs > lda? lda-j : bs);
    const int K = (k+bs > lda? lda-k : bs);

    for (int kk = 0; kk < K; kk++) {
        memcpy((void *) (AA + (kk * M)), (const void *) (A + i + (k + kk)*lda), M * sizeof(double));
    }
    for (int jj = 0; jj < N; jj++) {
        memcpy((void *) (BB + (jj * K)), (const void *) (B + k + (j + jj)*lda), K * sizeof(double));
        memcpy((void *) (CC + (jj * M)), (const void *) (C + i + (j + jj)*lda), M * sizeof(double));
    }

    if (bs == BLOCK_SIZE_2) {
      square_dgemm_varblock(BLOCK_SIZE_1, bs, AA, BB, CC);
    } else {
      basic_dgemm_copied(M, N, K, AA, BB, CC);
    }

    for (int jj = 0; jj < N; jj++) {
        memcpy((void *) (C + i + (j + jj)*lda), (const void *) (CC + (jj * M)), M * sizeof(double));
    }
}

void square_dgemm_varblock(const int bs,
  const int MM, const double *restrict A, const double *restrict B, double *restrict C)
{
    const double *AA = (double *) _mm_malloc(bs * bs * sizeof(double), BYTE_ALIGNMENT);
    const double *BB = (double *) _mm_malloc(bs * bs * sizeof(double), BYTE_ALIGNMENT);
    double *CC = (double *) _mm_malloc(bs * bs * sizeof(double), BYTE_ALIGNMENT);
    const int n_blocks = MM / bs + (MM%bs? 1 : 0);
    int bi, bj, bk;
    for (bk = 0; bk < n_blocks; ++bk) {
        const int k = bk * bs;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * bs;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * bs;
                if (bs == BLOCK_SIZE_1 && (i+bs > MM || j+bs > MM || k+bs > MM)) {
                  const int M = (i+bs > MM? MM-i : bs);
                  const int N = (j+bs > MM? MM-j : bs);
                  const int K = (k+bs > MM? MM-k : bs);
                  const double *AAA = (double *) _mm_malloc(M * K * sizeof(double), BYTE_ALIGNMENT);
                  const double *BBB = (double *) _mm_malloc(K * N * sizeof(double), BYTE_ALIGNMENT);
                  double *CCC = (double *) _mm_malloc(M * N * sizeof(double), BYTE_ALIGNMENT);
                  do_block(bs, MM, AAA, BBB, CCC, A, B, C, i, j, k);
                  _mm_free((void *) AAA);
                  _mm_free((void *) BBB);
                  _mm_free((void *) CCC);
                } else {
                  do_block(bs, MM, AA, BB, CC, A, B, C, i, j, k);
                }
            }
        }
    }
    _mm_free((void *) AA);
    _mm_free((void *) BB);
    _mm_free((void *) CC);
}

void square_dgemm(const int MM, const double *restrict A, const double *restrict B, double *restrict C)
{
    int pad = MM % BLOCK_SIZE_2;
    if (pad == 0) {
      square_dgemm_varblock(BLOCK_SIZE_2, MM, A, B, C);
    } else {
      int paddedMM = MM - pad + BLOCK_SIZE_2;
      int paddedB = paddedMM * paddedMM * sizeof(double);
      int row = MM * sizeof(double);
      const double *AA = (double *) _mm_malloc(paddedB, BYTE_ALIGNMENT);
      const double *BB = (double *) _mm_malloc(paddedB, BYTE_ALIGNMENT);
      double *CC = (double *) _mm_malloc(paddedB, BYTE_ALIGNMENT);
      memset((void *) AA, 0, paddedB); memset((void *) BB, 0, paddedB); memset((void *) CC, 0, paddedB);
      for (int i = 0; i < MM; i++) {
        memcpy((void *) (AA + (i * paddedMM)), (const void *) (A + i * MM), row);
        memcpy((void *) (BB + (i * paddedMM)), (const void *) (B + i * MM), row);
      }
      square_dgemm_varblock(BLOCK_SIZE_2, paddedMM, AA, BB, CC);
      for (int i = 0; i < MM; i++) {
        memcpy((void *) (C + (i * MM)), (const void *) (CC + i * paddedMM), row);
      }
    }
}
