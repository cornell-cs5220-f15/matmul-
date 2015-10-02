#include "immintrin.h"

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

#define KERNEL_SIZE 8

#ifndef NULL
#define NULL ((void *)0)
#endif

double *A_KERNEL = NULL;
double *B_KERNEL = NULL;
double *C_KERNEL = NULL;

// more convenient access; column major
#define A(i, j) A[(j)*M + (i)]
#define B(i, j) B[(j)*M + (i)]
#define C(i, j) C[(j)*M + (i)]

// more convenient access; row major
#define A_KERNEL(i, j) A_KERNEL[i*KERNEL_SIZE + j]
#define B_KERNEL(i, j) B_KERNEL[i*KERNEL_SIZE + j]
#define C_KERNEL(i, j) C_KERNEL[i*KERNEL_SIZE + j]

// assumes zmm08-15 already have the rows of B, and that zmm00-07 can be clobbered
inline void row8x8(uint row, double * restrict A, double * restrict C,
                   __m512d zmm00, __m512d zmm01, __m512d zmm02, __m512d zmm03,
                   __m512d zmm04, __m512d zmm05, __m512d zmm06, __m512d zmm07,
                   __m512d zmm08, __m512d zmm09, __m512d zmm10, __m512d zmm11,
                   __m512d zmm12, __m512d zmm13, __m512d zmm14, __m512d zmm15) {

    __assume_aligned(A, BYTE_ALIGN);// unsure if this is necessary with an inline being called
    __assume_aligned(C, BYTE_ALIGN);// by an inline, but shouldn't cause too much of a ruckus
    
    // Broadcast each element of Matrix A Row 1 into a zmm register
    // note that there is no _mm512_broadcast_sd; but this works too ;)
    zmm00 = _mm512_set1_pd(A[row*8 + 0]); zmm01 = _mm512_set1_pd(A[row*8 + 1]);
    zmm02 = _mm512_set1_pd(A[row*8 + 2]); zmm03 = _mm512_set1_pd(A[row*8 + 3]);
    zmm04 = _mm512_set1_pd(A[row*8 + 4]); zmm05 = _mm512_set1_pd(A[row*8 + 5]);
    zmm06 = _mm512_set1_pd(A[row*8 + 6]); zmm07 = _mm512_set1_pd(A[row*8 + 7]);

    // Multiply each element of A Row 1 with each Row of B
    zmm00 = _mm512_mul_pd(zmm00, zmm08); zmm01 = _mm512_mul_pd(zmm01, zmm09);
    zmm02 = _mm512_mul_pd(zmm02, zmm10); zmm03 = _mm512_mul_pd(zmm03, zmm11);
    zmm04 = _mm512_mul_pd(zmm04, zmm12); zmm05 = _mm512_mul_pd(zmm05, zmm13);
    zmm06 = _mm512_mul_pd(zmm06, zmm14); zmm07 = _mm512_mul_pd(zmm07, zmm15);

    // Add up partial sums to reduce from 8 separate to 4 separate [read left right from previous]
    zmm00 = _mm512_add_pd(zmm00, zmm01); zmm02 = _mm512_add_pd(zmm02, zmm03);
    zmm04 = _mm512_add_pd(zmm04, zmm05); zmm06 = _mm512_add_pd(zmm06, zmm07);

    // Add up partial sums to reduce from 4 separate to 2 separate [read left right from previous]
    zmm00 = _mm512_add_pd(zmm00, zmm02); zmm04 = _mm512_add_pd(zmm04, zmm06);

    // Add up partial sums to reduce from 2 separate to final value [read left right from previous]
    zmm00 = _mm512_add_pd(zmm00, zmm04);

    // zmm00 now holds the entire row computation for C, store it back
    _mm512_store_pd((double *) (C + row*8));
}

inline void vectorized8x8(double * restrict A, double * restrict B, double * restrict C) {
    __assume_aligned(A, BYTE_ALIGN);
    __assume_aligned(B, BYTE_ALIGN);
    __assume_aligned(C, BYTE_ALIGN);
    
    // adapted from:
    //     https://software.intel.com/en-us/articles/benefits-of-intel-avx-for-small-matrices

    __m512d zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,// these will store A
            zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;// these will store B

    // Read in the 8 rows of Matrix B into zmm registers
    zmm08 = _mm512_load_pd((double *) (B + 0*8)); zmm09 = _mm512_load_pd((double *) (B + 1*8));
    zmm10 = _mm512_load_pd((double *) (B + 2*8)); zmm11 = _mm512_load_pd((double *) (B + 3*8));
    zmm12 = _mm512_load_pd((double *) (B + 4*8)); zmm13 = _mm512_load_pd((double *) (B + 5*8));
    zmm14 = _mm512_load_pd((double *) (B + 6*8)); zmm15 = _mm512_load_pd((double *) (B + 7*8));

    // row by row computations
    row8x8(0, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(1, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(2, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(3, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(4, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(5, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(6, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    row8x8(7, A, C,
           zmm00, zmm01, zmm02, zmm03, zmm04, zmm05, zmm06, zmm07,
           zmm08, zmm09, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15);
    
}

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N
  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    // int i, j, k;
    // for (j = 0; j < N; ++j) {
    //     for (k = 0; k < K; ++k){
    //         double bkj = B[j*lda+k];
    //         for (i = 0; i < M; ++i) {
    //             C[j*lda+i] += A[k*lda+i] * bkj;
    //         }
    //     }
    // }
    int i, j, k;
    for (j = 0; j < N; ++j) {
        // for (k = 0; k < K; ++k){
        for (i = 0; i < M; ++i) {
            // copy to aligned memory
            #pragma unroll
            for(int col = 0; col < KERNEL_SIZE; ++col) {
                for(int row = 0; row < KERNEL_SIZE; ++row) {
                    A_KERNEL(row, col) = A(row, col);
                    B_KERNEL(row, col) = B(row, col*lda);
                }
            }

            // double bkj = B[j*lda+k];
            // for (i = 0; i < M; ++i) {
            //     C[j*lda+i] += A[k*lda+i] * bkj;
            // }
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

    A_KERNEL = (double *) _mm_malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double), BYTE_ALIGN);// BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGN);
    B_KERNEL = (double *) _mm_malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double), BYTE_ALIGN);// BLOCK_SIZE * BLOCK_SIZE * sizeof(double), BYTE_ALIGN);
    C_KERNEL = (double *) _mm_malloc(KERNEL_SIZE               * sizeof(double), BYTE_ALIGN);// BLOCK_SIZE              * sizeof(double), BYTE_ALIGN);
    
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

    _mm_free(A_KERNEL); A_KERNEL = NULL;
    _mm_free(B_KERNEL); B_KERNEL = NULL;
    _mm_free(C_KERNEL); C_KERNEL = NULL;
}
