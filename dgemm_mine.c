const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64) // Multiples of 8 bytes!
#endif

void block_dgemm(const double *restrict A, 
                 const double *restrict B, 
                 double *restrict C)
{
    int i, j, k;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        for (k = 0; k < BLOCK_SIZE; ++k) {
            double bkj = B[j*BLOCK_SIZE + k];
            for (i = 0; i < BLOCK_SIZE; ++i) {
                C[j*BLOCK_SIZE + i] += A[k*BLOCK_SIZE + i] * bkj;
            }
        }
    }
}

// void pad_square_matrix(const int oldSize, const int newSize, 
// 	                   const double *A, double *A_new)
// {
// 	int i, j;
// 	for (i = 0; i < newSize; ++i) {
// 		for (j = 0; j < newSize; ++j) {
// 			if (i < oldSize && j < oldSize) {
// 				A_new[j*newSize + i] = A[j*oldSize + i];
// 			}
// 			else {
// 				A_new[j*newSize + i] = 0;
// 			}
// 		}
// 	}
// }

void realloc_block(const int lda1, const int lda2,
                   const double *A1, double *A2, 
                   const int limit)//const int r_limit, const int c_limit) 
{
    int i, j;
    for (i = 0; i < lda2; ++i) {
        for (j = 0; j < lda2; ++j) {
            A2[j*lda2 + i] = (i*j < limit) ? A1[j*lda1 + i] : 0;
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    double A_temp[BLOCK_SIZE * BLOCK_SIZE]; //__attribute__ ((aligned (64)));
    double B_temp[BLOCK_SIZE * BLOCK_SIZE];
    double C_temp[BLOCK_SIZE * BLOCK_SIZE];

    /*
        A is M-by-K
        B is K-by-N
        C is M-by-N

        lda is the leading dimension of the matrix (the M of square_dgemm).
    */
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    // basic_dgemm(lda, M, N, K,
    //             A + i + k*lda, B + k + j*lda, C + i + j*lda);

    realloc_block(lda, BLOCK_SIZE, A + i + k*lda, A_temp, M*K);
    realloc_block(lda, BLOCK_SIZE, B + k + j*lda, B_temp, K*N);
    realloc_block(lda, BLOCK_SIZE, C + i + j*lda, C_temp, M*N);

    block_dgemm(A_temp, B_temp, C_temp);

    // Copy results from C_temp back to C
    // double *C_block = C + i + j*lda;
    int ci, cj;
    for (ci = 0; ci < M; ++ci) {
        for (cj = 0; cj < N; ++cj) {
            C[(j + cj)*lda + (i+ci)] = C_temp[cj*BLOCK_SIZE + ci];
        }
    }
}

// void square_dgemm(const int M, const double *A, const double *B, double *C)
// {
//     // Round up to nearest multiple of BLOCK_SIZE
// 	const int newSize = M + BLOCK_SIZE - 1 - (M - 1) % BLOCK_SIZE;

// 	double* A_new = (double*) malloc(newSize * newSize * sizeof(double));
//     double* B_new = (double*) malloc(newSize * newSize * sizeof(double));
//     double* C_new = (double*) malloc(newSize * newSize * sizeof(double));
//     pad_square_matrix(M, newSize, A, A_new);
//     pad_square_matrix(M, newSize, B, B_new);
//     pad_square_matrix(M, newSize, C, C_new);

//     const int n_blocks = newSize / BLOCK_SIZE;
//     int bi, bj, bk;
//     for (bi = 0; bi < n_blocks; ++bi) {
//         const int i = bi * BLOCK_SIZE;
//         for (bj = 0; bj < n_blocks; ++bj) {
//             const int j = bj * BLOCK_SIZE;
//             for (bk = 0; bk < n_blocks; ++bk) {
//                 const int k = bk * BLOCK_SIZE;
//                 do_block( newSize, 
//                           A_new + i + k*newSize, 
//                           B_new + k + j*newSize, 
//                           C_new + i + j*newSize );
//             }
//         }
//     }

//     // Copy results from C_new back into C...
//     int i, j;
//     for (i = 0; i < M; ++i) {
// 		for (j = 0; j < M; ++j) {
// 			C[j*M + i] = C_new[j*newSize + i];
// 		}
// 	}
// }

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
