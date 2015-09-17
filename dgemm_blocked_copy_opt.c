#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

#define COPY_OPT_THRESH 128

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{

    int i, j, k, lda_eff;
    double cij;

    /* if(copy_optimization) { */
        /* double* restrict A_copy; */
        /* double* restrict B_copy; */
        /* A_copy = (double*) malloc( sizeof(double) * M * K); */
        /* for(k=0; k<K; k++) { */
            /* memcpy(A_copy + k*M, A + k*lda, M); */
        /* } */
        /* B_copy = (double*) malloc( sizeof(double) * N * K); */
        /* for(j=0; j<N; j++) { */
            /* memcpy(B_copy + j*K, B + j*lda,K); */
        /* } */

        /* double diff; */
        /* for(i=0; i<M; i++) { */
            /* for(k=0; k<N; k++) { */
                /* diff = (A[k*lda + i] - A_copy[k*M + i]); */
                /* printf("%f\n",diff); */
                /* if( abs(diff) > 0.0000001) { */
                    /* printf("%d %d %f %f %f\n",k, i, A[k*lda + i], A_copy[k*M + i], abs(diff)); */
                /* } */
            /* } */
        /* } */

        /* for(j=0; j<N; j++) { */
            /* for(k=0; k<N; k++) { */
                /* printf("%f\n",diff); */
                /* diff = (B[j*lda + k] - B_copy[j*K + k]); */
                /* if( abs(diff) > 0.0000001) { */
                    /* printf("%d %d %f %f %f\n",k, i, B[j*lda + j], B_copy[j*K + k], abs(diff)); */
                /* } */
            /* } */
        /* } */

        /* for (j = 0; j < N; ++j) { */
            /* for (i = 0; i < M; ++i) { */
                /* cij = C[j*lda+i]; */
                /* for (k = 0; k < K; ++k) { */
                    /* cij += A_copy[k*M+i] * B_copy[j*K+k]; */
                /* } */
                /* C[j*lda+i] = cij; */
            /* } */
        /* } */
    
        /* free(A_copy); */
        /* free(B_copy); */
    /* } else { */
        for (j = 0; j < N; ++j) {
            for (i = 0; i < K; ++i) {
                cij = C[j*lda+i];
                for (k = 0; k < M; ++k) {
                    cij += A[i*lda+k] * B[j*lda+k];
                }
                C[j*lda+i] = cij;
            }
        }
    /* } */
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
    double *A_copy;
    int i,j;
    // Turn on copy optimization after a certain threshold matrix size
    if(M > COPY_OPT_THRESH) {
        //Make A transpose the first matrix.
        //Flip the multiplication order.
        //No more cache misses! Yay :D
        A_copy = (double*) malloc( sizeof(double) * M * M );
        for(i = 0; i<M; i++) {
            for(j = 0; j<M; j++) {
                A_copy[i * M + j] = A[j * M + i];
            }
        }
        
    }

    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                do_block(M, A_copy, B, C, i, j, k);
            }
        }
    }
}

