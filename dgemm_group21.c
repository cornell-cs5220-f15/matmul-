#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

#define ALIGNMENT ((int) 64)

const char* dgemm_desc = "Basic, three-loop dgemm with copy optimization";

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
    __assume_aligned(A, 64);
    __assume_aligned(B, 64);
    __assume_aligned(C, 64);
    for (j = 0; j < N; ++j) {
        jlda = j*lda;
        __assume_aligned(C, 64);
        for (i = 0; i < M; ++i) {
            ilda = i*lda;
            double cij = C[jlda+i];
            __assume_aligned(A, 64);
            __assume_aligned(B, 64);
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

double* padded_transpose(const double* A, const int M) {
    int i,j;
    const int M_padded = M + ALIGNMENT - M%ALIGNMENT;
    double* A_copy = (double*) _mm_malloc( sizeof(double) * M_padded * M_padded, 64 );

    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[i * M_padded + j] = A[j * M + i];
        }
    }
    
    for(i=M; i<M_padded; i++) {
        for(j=M; j<M_padded; j++) {
            A_copy[i * M_padded + j] = 0;
        }
    }

    return A_copy;
}

double* padded_copy(const double* A, const int M) {
    int i,j;
    const int M_padded = M + ALIGNMENT - M%ALIGNMENT;
    double* A_copy = (double*) _mm_malloc( sizeof(double) * M_padded * M_padded, 64 );

    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[j * M_padded + i] = A[j * M + i];
        }
    }
    
    for(i=M; i<M_padded; i++) {
        for(j=M; j<M_padded; j++) {
            A_copy[j * M_padded + i] = 0;
        }
    }

    return A_copy;
}

void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
    int i,j,k;
    if( M <= 1024 ) {
        double *A_copy;

        A_copy = (double*) malloc( sizeof(double) * M * M );
        for(i = 0; i<M; i++) {
            for(j = 0; j<M; j++) {
                A_copy[i * M + j] = A[j * M + i];
            }
        }
            
        for (j = 0; j < M; ++j) {
            for (i = 0; i < M; ++i) {
                double cij = C[j*M+i];
                for (k = 0; k < M; ++k)
                    cij += A_copy[i*M+k] * B[j*M+k];
                C[j*M+i] = cij;
            }
        }

        free(A_copy);
    } else {
        int bi, bj, bk;
        const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
        double* A_copy = padded_transpose(A, M);
        double* B_copy = padded_copy(B, M);
        double* C_copy = padded_copy(C, M);
        const int M_padded = M + ALIGNMENT - M%ALIGNMENT;
        for (bi = 0; bi < n_blocks; ++bi) {
            const int i = bi * BLOCK_SIZE;
            for (bj = 0; bj < n_blocks; ++bj) {
                const int j = bj * BLOCK_SIZE;
                for (bk = 0; bk < n_blocks; ++bk) {
                    const int k = bk * BLOCK_SIZE;
                    do_block(M_padded, A_copy, B_copy, C_copy, i, j, k);
                }
            }
        }
        for(i = 0; i<M; i++) {
            for(j = 0; j<M; j++) {
                C[j * M + i] = C_copy[j * M_padded + i];
            }
        }

        _mm_free(A_copy);
        _mm_free(B_copy);
        _mm_free(C_copy);
    }
}

