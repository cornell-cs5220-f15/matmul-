#include <stdio.h>
#include <stdlib.h>

const char* dgemm_desc = "Blocked dgemm with copy optimization.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    int jlda, ilda;
    for (j = 0; j < N; ++j) {
        jlda = j*lda;
        for (i = 0; i < M; ++i) {
            ilda = i*lda;
            double cij = C[jlda+i];
            for (k = 0; k < K; ++k) {
                cij += A[ilda+k] * B[jlda+k];
            }
            C[jlda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + k + i*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    double* A_copy;
    int i, j, k;
    A_copy = (double*)malloc( sizeof(double) * M * M, 64 );

    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[i * M + j] = A[j * M + i];
        }
    }

    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A_copy, B, C, i, j, k);
            }
        }
    }

    free(A_copy);
}

