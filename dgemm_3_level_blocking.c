#include <stdlib.h>
#include <string.h>

#include "copy.h"
#include "indexing.h"
#include "transpose.h"

const char* dgemm_desc = "mjw297 dgemm.";

#ifndef BLOCK_SIZE_L1
#define BLOCK_SIZE_L1 ((int) 32)
#endif
#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 ((int) 128)
#endif
#ifndef BLOCK_SIZE_L3
#define BLOCK_SIZE_L3 ((int) 1024)
#endif

double A_[BLOCK_SIZE_L1 * BLOCK_SIZE_L1];
double B_[BLOCK_SIZE_L1 * BLOCK_SIZE_L1];

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

    // clear A_ and B_
    memset(A_, 0, BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double));
    memset(B_, 0, BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double));

    // transpose A into A_
    cm_transpose_into(A, lda, lda, M, K, A_, BLOCK_SIZE_L1, BLOCK_SIZE_L1);

    // copy B into B_
    cm_copy_into(B, lda, lda, K, N, B_, BLOCK_SIZE_L1, BLOCK_SIZE_L1);

    for (j = 0; j < BLOCK_SIZE_L1; ++j) {
        for (i = 0; i < BLOCK_SIZE_L1; ++i) {
            double cij = C[cm(lda, lda, i, j)];
            for (k = 0; k < BLOCK_SIZE_L1; ++k) {
                cij += A_[rm(BLOCK_SIZE_L1, BLOCK_SIZE_L1, i, k)] * B_[cm(BLOCK_SIZE_L1, BLOCK_SIZE_L1, k, j)];
            }
            C[cm(lda, lda, i, j)] = cij;
        }
    }
}

void do_block_L1(const int lda,
              const double* restrict A,
              const double* restrict B,
                    double* restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE_L1 > lda? lda-i : BLOCK_SIZE_L1);
    const int N = (j+BLOCK_SIZE_L1 > lda? lda-j : BLOCK_SIZE_L1);
    const int K = (k+BLOCK_SIZE_L1 > lda? lda-k : BLOCK_SIZE_L1);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void do_block_L2(const int lda,
                 const double* A,
                 const double* restrict B,
                       double* restrict C,
                 const int i, const int j, const int k) {

    const int M = ((i+BLOCK_SIZE_L2 > lda? lda-i : BLOCK_SIZE_L2));
    const int m_blocks = M / BLOCK_SIZE_L1 + (M%BLOCK_SIZE_L1 ? 1 : 0);
    const int N = ((j+BLOCK_SIZE_L2 > lda? lda-j : BLOCK_SIZE_L2));
    const int n_blocks = N / BLOCK_SIZE_L1 + (N%BLOCK_SIZE_L1 ? 1 : 0);
    const int K = ((k+BLOCK_SIZE_L2 > lda? lda-k : BLOCK_SIZE_L2));
    const int k_blocks = K / BLOCK_SIZE_L1 + (K%BLOCK_SIZE_L1 ? 1 : 0);

    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j_L2 = j + bj * BLOCK_SIZE_L1;
        for (bi = 0; bi < m_blocks; ++bi) {
            const int i_L2 = i + bi * BLOCK_SIZE_L1;
            for (bk = 0; bk < k_blocks; ++bk) {
                const int k_L2 = k + bk * BLOCK_SIZE_L1;
                do_block_L1(lda, A, B, C, i_L2, j_L2, k_L2);
            }
        }
    }
}

void do_block_L3(const int lda,
                 const double* restrict A,
                 const double* restrict B,
                       double* restrict C,
                 const int i, const int j, const int k) {

    const int M = ((i+BLOCK_SIZE_L3 > lda? lda-i : BLOCK_SIZE_L3));
    const int m_blocks = M / BLOCK_SIZE_L2 + (M%BLOCK_SIZE_L2 ? 1 : 0);
    const int N = ((j+BLOCK_SIZE_L3 > lda? lda-j : BLOCK_SIZE_L3));
    const int n_blocks = N / BLOCK_SIZE_L2 + (N%BLOCK_SIZE_L2 ? 1 : 0);
    const int K = ((k+BLOCK_SIZE_L3 > lda? lda-k : BLOCK_SIZE_L3));
    const int k_blocks = K / BLOCK_SIZE_L2 + (K%BLOCK_SIZE_L2 ? 1 : 0);

    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j_L3 = j + bj * BLOCK_SIZE_L2;
        for (bi = 0; bi < m_blocks; ++bi) {
            const int i_L3 = i + bi * BLOCK_SIZE_L2;
            for (bk = 0; bk < k_blocks; ++bk) {
                const int k_L3 = k + bk * BLOCK_SIZE_L2;
                do_block_L2(lda, A, B, C, i_L3, j_L3, k_L3);
            }
        }
    }
}

void square_dgemm(const int M,
                  const double* restrict A,
                  const double* restrict B,
                  double* restrict C)
{
    int i, j, bi, bj, bk;
    // double* D = (double*) malloc(M * M * sizeof(double));

    // copy data from A->D to align memory accesses
    // i = column
    // j = row
    // for (i = 0; i < M; ++i) {
    //     for (j = 0; j < M; ++j) {
    //         D[j*M + i] = A[i*M + j];
    //     }
    // }

    const int n_blocks = M / BLOCK_SIZE_L2 + (M%BLOCK_SIZE_L2? 1 : 0);
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE_L2;
        for (bi = 0; bi < n_blocks; ++bi) {
            const int i = bi * BLOCK_SIZE_L2;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE_L2;
                do_block_L2(M, A, B, C, i, j, k);
            }
        }
    }
    // free(D);
}
