const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif


// void block_dgemm(const double *A, const double *B, double *C)
// {
//     int i, j, k;
//     for (j = 0; j < N; ++j) {
//     	for (k = 0; k < K; ++k) {
//     		double bkj = B[j*lda+k];
//     		for (i = 0; i < M; ++i) {
//     			C[j*lda + i] += A[k*lda + i] * bkj;
//     		}
//     	}
//     }
// }

void basic_dgemm(const int lda, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        for (k = 0; k < BLOCK_SIZE; ++k) {
            double bkj = B[j*lda+k];
            for (i = 0; i < BLOCK_SIZE; ++i) {
                C[j*lda + i] += A[k*lda + i] * bkj;
            }
        }
    }
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
    // Round up to nearest multiple of BLOCK_SIZE
	const int newSize = M + BLOCK_SIZE - 1 - (M - 1) % BLOCK_SIZE;

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
                basic_dgemm( newSize, 
                             A_new + i + k*newSize, 
                             B_new + k + j*newSize, 
                             C_new + i + j*newSize );
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
