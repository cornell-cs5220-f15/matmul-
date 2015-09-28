#include<stdlib.h>

const char* dgemm_desc = "dgemm with all the fixins";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 384)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N
  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A,
                 const double* restrict B,
                 double* restrict C)
{
    int i, j, k;

    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[i*lda+k] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double* restrict A,
              const double* restrict B,
              double* restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i*lda + k, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M,
                  const double* restrict A,
                  const double* restrict B,
                  double* restrict C)
{
    double* D = (double*) malloc(M * M * sizeof(double));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            D[j*M + i] = A[i*M + j];
        }
    }

    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bj = 0; bj < n_blocks; ++bj) {
                const int j = bj * BLOCK_SIZE;
                do_block(M, D, B, C, i, j, k);
            }
        }
    }
}
