#include <stdlib.h>
#include <string.h>

#include "clear.h"
#include "copy.h"
#include "indexing.h"
#include "transpose.h"

const char* dgemm_desc = "Padded blocked matmul with conditionals.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

double A_[BLOCK_SIZE * BLOCK_SIZE];
double B_[BLOCK_SIZE * BLOCK_SIZE];

/*
 * A is M-by-K
 * B is K-by-N
 * C is M-by-N
 *
 * lda is the leading dimension of the matrix (the M of square_dgemm).
 */
void basic_dgemm(const int lda,
                 const int M, const int N, const int K,
                 const double* A,
                 const double* B,
                       double* C) {
    int i, j, k;

    // transpose A into A_
    cm_transpose_into(A, lda, lda, M, K, A_, BLOCK_SIZE, BLOCK_SIZE);

    // copy B into B_
    cm_copy_into(B, lda, lda, K, N, B_, BLOCK_SIZE, BLOCK_SIZE);

    if (M == BLOCK_SIZE && N == BLOCK_SIZE && K == BLOCK_SIZE) {
        for (j = 0; j < BLOCK_SIZE; ++j) {
            for (i = 0; i < BLOCK_SIZE; ++i) {
                double cij = C[cm(lda, lda, i, j)];
                for (k = 0; k < BLOCK_SIZE; ++k) {
                    cij += A_[rm(BLOCK_SIZE, BLOCK_SIZE, i, k)] * B_[cm(BLOCK_SIZE, BLOCK_SIZE, k, j)];
                }
                C[cm(lda, lda, i, j)] = cij;
            }
        }
    } else {
        for (j = 0; j < N; ++j) {
            for (i = 0; i < M; ++i) {
                double cij = C[cm(lda, lda, i, j)];
                for (k = 0; k < K; ++k) {
                    cij += A_[rm(BLOCK_SIZE, BLOCK_SIZE, i, k)] * B_[cm(BLOCK_SIZE, BLOCK_SIZE, k, j)];
                }
                C[cm(lda, lda, i, j)] = cij;
            }
        }
    }
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
