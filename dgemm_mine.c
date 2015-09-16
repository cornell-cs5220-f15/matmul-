const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif


void basic_dgemm(const int lda, const int M, const int N, const int K,
	             const double *A, const double *B, double *C)
{
    int i, j, k;
    for (j = 0; j < N; ++j) {
    	for (k = 0; k < K; ++k) {
    		double bkj = B[j*lda+k];
    		for (i = 0; i < M; ++i) {
    			C[j*lda + i] += A[k*lda + i] * bkj;
    		}
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
    basic_dgemm(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void pad_square_matrix(const int oldSize, const int newSize, 
	                   const double *A, double *A_new)
{
	int i, j;
	for (i = 0; i < newSize; ++i) {
		for (j = 0; j < newSize; ++j) {
			if (i < oldSize && j < oldSize) {
				A_new[j*newSize + i] = A[j*oldSize + i];
			}
			else {
				A_new[j*newSize + i] = 0;
			}
		}
	}
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

	// const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
 //    int bi, bj, bk;
 //    for (bi = 0; bi < n_blocks; ++bi) {
 //        const int i = bi * BLOCK_SIZE;
 //        for (bj = 0; bj < n_blocks; ++bj) {
 //            const int j = bj * BLOCK_SIZE;
 //            for (bk = 0; bk < n_blocks; ++bk) {
 //                const int k = bk * BLOCK_SIZE;
 //                do_block(M, A, B, C, i, j, k);
 //            }
 //        }
 //    }

	const int padSize = M % BLOCK_SIZE;
	const int newSize = M + padSize;

	double* A_new = (double*) malloc(newSize * newSize * sizeof(double));
    double* B_new = (double*) malloc(newSize * newSize * sizeof(double));
    double* C_new = (double*) malloc(newSize * newSize * sizeof(double));
    pad_square_matrix(M, newSize, A, A_new);
    pad_square_matrix(M, newSize, B, B_new);
    pad_square_matrix(M, newSize, C, C_new);

    const int n_blocks = newSize / BLOCK_SIZE;
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(newSize, A_new, B_new, C_new, i, j, k);
            }
        }
    }

    // Copy results from C_new back into C...
    int i, j;
    for (i = 0; i < M; ++i) {
		for (j = 0; j < M; ++j) {
			C[j*M + i] = C_new[j*newSize + i];
		}
	}
}
