const char* dgemm_desc = "Simple blocked dgemm.";

/* In theory, we will need to fit all of A and all of B into the L2 cache,
 * as well as a 1-D vector of BLOCK_SIZE for C.
 *
 * Given the 256KB L2 cache size for the cluster, we must compute
 *
 *     (256 * 1024)bytes = 2b^2 + b, where b is BLOCK_SIZE
 *                       ~ 361
 *
 * However, having byte-aligned memory is desireable in order to compute
 * using various vector operations.  The result being that we want to
 * take into consideration the byte-alignment we seek to use here.
 *
 * So if we choose a byte alignment of 64, then we should choose BLOCK_SIZE
 * to be floor(361 / 64) * 64 = 320.  However, if we seek to use a 16byte
 * alignment, then we should choose BLOCK_SIZE = floor(361 / 16) * 16 = 352.
 */

#define BYTE_ALIGN 64
#define BLOCK_SIZE ((361 / BYTE_ALIGN) * BYTE_ALIGN) // important to leave as a define for compiler optimizations

#ifndef NULL
#define NULL ((void *)0)
#endif

double *A_BLOCK = NULL;
double *B_BLOCK = NULL;
double *C_BLOCK = NULL;

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N
  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k){
            double bkj = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                C[j*lda+i] += A[k*lda+i] * bkj;
            }
        }
    }
}

#include <string.h>

void do_block(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    
    // copy over to aligned memory
    register int a_start = lda*i;
    for(int m = 0; m < BLOCK_SIZE; ++m) {
        if(m >= M) {
            A_BLOCK[m] = 0.0;
        }
        else {
            A_BLOCK[m] = A[a_start + m];
        }
    }

    register int b_start = lda*j;
    for(int n = 0; n < BLOCK_SIZE; ++n) {
        if(n >= N) {
            B_BLOCK[n] = 0.0;
        }
        else {
            B_BLOCK[n] = B[b_start + n];
        }
    }

    register int c_start = lda*k;
    for(int kk = 0; k < BLOCK_SIZE; ++k) {
        if(k >= K) {
            C_BLOCK[
        }
    }


    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

#include <stdlib.h>

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    if (M <= BLOCK_SIZE) {
       basic_dgemm(M, M, M, M, A, B, C);
       return;
    }

    A_BLOCK = (double *) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGN);
    B_BLOCK = (double *) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGN);
    C_BLOCK = (double *) _mm_malloc(BLOCK_SIZE              * sizeof(double), BYTE_ALIGN);
    
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

    _mm_free(A_BLOCK); A_BLOCK = NULL;
    _mm_free(B_BLOCK); B_BLOCK = NULL;
    _mm_free(C_BLOCK); C_BLOCK = NULL;
}
