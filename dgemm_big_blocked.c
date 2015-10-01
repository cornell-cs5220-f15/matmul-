#include <stdlib.h>

#include "copy.h"
#include "indexing.h"
#include "transpose.h"

const char* dgemm_desc = "Large blocked matmul with A tranposed in blocks.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 1024)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda,
                 const int M, const int N, const int K,
                 const double* A,
                 const double* B,
                       double* C) {
    int i, j, k;
    double *A_ = cm_transpose(A, lda, lda, M, K);

    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            double cij = C[cm(lda, lda, i, j)];
            for (k = 0; k < K; ++k) {
                cij += A_[rm(M, K, i, k)] * B[cm(lda, lda, k, j)];
            }
            C[cm(lda, lda, i, j)] = cij;
        }
    }

    free(A_);
}

void do_block(const int lda,
              const double* A,
              const double* B,
                    double* C,
              const int i, const int j, const int k) {
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                &A[cm(lda, lda, i, k)], &B[cm(lda, lda, k, j)], &C[cm(lda, lda, i, j)]);
}

void square_dgemm(const int M,
                  const double* A,
                  const double* B,
                        double* C) {
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                const int i = bi * BLOCK_SIZE;
                const int j = bj * BLOCK_SIZE;
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
