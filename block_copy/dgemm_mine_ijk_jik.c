#include <stdio.h>
#include <stdlib.h>

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C,
                 double *aa, double *bb, double *cc)
{
    int i, j, k;

    for (k = 0; k < K; ++k){
      for(i = 0; i < M; ++i){       
        aa[i*BLOCK_SIZE + k] = A[k*lda + i];
      }
    }

    for(j = 0; j < N; ++j){
      for(k = 0; k < K; ++k){
        bb[j*BLOCK_SIZE+k] = B[j*lda+k];
      }
    }

    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += aa[i*BLOCK_SIZE+k] * bb[j*BLOCK_SIZE+k];
            }
            C[j*lda+i] = cij;
        }
    }

}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              double *aa, double *bb, double *cc,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda, aa, bb, cc);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

    //double* t = (double*)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double aa[BLOCK_SIZE * BLOCK_SIZE] = {0};
    double bb[BLOCK_SIZE * BLOCK_SIZE] = {0};
    double cc[BLOCK_SIZE * BLOCK_SIZE] = {0};

    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, aa, bb, cc, i, j, k);
            }
        }
    }
}


/*
    for(i = 0; i < M; ++i){
      for(j = 0; j < M; ++j){
        printf("%f ", A[i*M + j]);
      }
      printf("\n");
      
    }

    printf("\n\n");
    for(i = 0; i < M; ++i){
      for(j = 0; j < M; ++j){
        printf("%f ", a[i*M + j]);
      }
      printf("\n");
    }
*/


/*
void square_dgemm(const int M, const double *A, const double *B, double *C)
{    
    int i, j, k;
    double* a = (double*) malloc(M * M * sizeof(double));

    for(i = 0; i < M; ++i){
      for(j = 0; j <= i; ++j){
        a[i*M+j] = A[j*M+i];
        a[j*M+i] = A[i*M+j];
      }
    }


    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += a[i*M+k] * B[j*M + k];
                //cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
}
*/
