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

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)

double * restrict A_KERNEL = NULL;
double * restrict B_KERNEL = NULL;
double * restrict C_KERNEL = NULL;

// more convenient access; column major
#define A(i, j) A[(j) + (i)]
#define B(i, j) B[(j) + (i)]
#define C(i, j) C[(j) + (i)]

// more convenient access; row major
#define A_KERNEL(i, j) A_KERNEL[i*KERNEL_SIZE + j]
#define B_KERNEL(i, j) B_KERNEL[i*KERNEL_SIZE + j]
#define C_KERNEL(i, j) C_KERNEL[i*KERNEL_SIZE + j]

// assumes zmm16-31 already have the rows of B, and that ymm00-015 can be clobbered
inline void row8x8(unsigned int row, double * restrict A, double * restrict C,
                   __m256d ymm00, __m256d ymm01, __m256d ymm02, __m256d ymm03, __m256d ymm04, __m256d ymm05, __m256d ymm06, __m256d ymm07,// piecewise store A
                   __m256d ymm08, __m256d ymm09, __m256d ymm10, __m256d ymm11, __m256d ymm12, __m256d ymm13, __m256d ymm14, __m256d ymm15,
                   __m256d ymm16, __m256d ymm17, __m256d ymm18, __m256d ymm19, __m256d ymm20, __m256d ymm21, __m256d ymm22, __m256d ymm23,// piecewise store B
                   __m256d ymm24, __m256d ymm25, __m256d ymm26, __m256d ymm27, __m256d ymm28, __m256d ymm29, __m256d ymm30, __m256d ymm31) {

    __assume_aligned(A, BYTE_ALIGN);// unsure if this is necessary with an inline being called
    __assume_aligned(C, BYTE_ALIGN);// by an inline, but shouldn't cause too much of a ruckus

    // Broadcast each element of Matrix A Row 1 into a ymm register
    // If row = [ a b c d e f g h ], then we need two registers for each
    
    //
    // process left half
    //
    // broadcast out
    ymm00 = _mm256_broadcast_sd(A + row*8 + 0); ymm01 = _mm256_broadcast_sd(A + row*8 + 0);// a
    ymm02 = _mm256_broadcast_sd(A + row*8 + 1); ymm03 = _mm256_broadcast_sd(A + row*8 + 1);// b
    ymm04 = _mm256_broadcast_sd(A + row*8 + 2); ymm05 = _mm256_broadcast_sd(A + row*8 + 2);// c
    ymm06 = _mm256_broadcast_sd(A + row*8 + 3); ymm07 = _mm256_broadcast_sd(A + row*8 + 3);// d

    // multiply
    ymm00 = _mm256_mul_pd(ymm00, ymm16); ymm01 = _mm256_mul_pd(ymm01, ymm17);// row 1
    ymm02 = _mm256_mul_pd(ymm02, ymm18); ymm03 = _mm256_mul_pd(ymm03, ymm19);// row 2
    ymm04 = _mm256_mul_pd(ymm04, ymm20); ymm05 = _mm256_mul_pd(ymm05, ymm21);// row 3
    ymm06 = _mm256_mul_pd(ymm06, ymm22); ymm07 = _mm256_mul_pd(ymm07, ymm23);// row 4

    // add up left half
    ymm00 = _mm256_add_pd(ymm00, ymm01); ymm02 = _mm256_add_pd(ymm02, ymm03);
    ymm04 = _mm256_add_pd(ymm04, ymm05); ymm06 = _mm256_add_pd(ymm06, ymm07);

    ymm00 = _mm256_add_pd(ymm00, ymm02); ymm04 = _mm256_add_pd(ymm04, ymm06);

    ymm00 = _mm256_add_pd(ymm00, ymm04);// ymm00 holds left half

    //
    // process right half
    //
    ymm08 = _mm256_broadcast_sd(A + row*8 + 4); ymm09 = _mm256_broadcast_sd(A + row*8 + 4);// e
    ymm10 = _mm256_broadcast_sd(A + row*8 + 5); ymm11 = _mm256_broadcast_sd(A + row*8 + 5);// f
    ymm12 = _mm256_broadcast_sd(A + row*8 + 6); ymm13 = _mm256_broadcast_sd(A + row*8 + 6);// g
    ymm14 = _mm256_broadcast_sd(A + row*8 + 7); ymm15 = _mm256_broadcast_sd(A + row*8 + 7);// h

    // multiply
    ymm08 = _mm256_mul_pd(ymm08, ymm24); ymm09 = _mm256_mul_pd(ymm09, ymm25);// row 5
    ymm10 = _mm256_mul_pd(ymm10, ymm26); ymm11 = _mm256_mul_pd(ymm11, ymm27);// row 6
    ymm12 = _mm256_mul_pd(ymm12, ymm28); ymm13 = _mm256_mul_pd(ymm13, ymm29);// row 7
    ymm14 = _mm256_mul_pd(ymm14, ymm30); ymm15 = _mm256_mul_pd(ymm15, ymm31);// row 8

    // add up right half
    ymm08 = _mm256_add_pd(ymm08, ymm09); ymm10 = _mm256_add_pd(ymm10, ymm11);
    ymm12 = _mm256_add_pd(ymm12, ymm13); ymm14 = _mm256_add_pd(ymm14, ymm15);

    ymm08 = _mm256_add_pd(ymm08, ymm10); ymm12 = _mm256_add_pd(ymm12, ymm14);

    ymm08 = _mm256_add_pd(ymm08, ymm12);// ymm08 holds right half

    // ym00 and ym08 now hold the left and right halves, store back in C
    _mm256_store_pd((double *) (C+row*8), ymm00); _mm256_store_pd((double *) (C+row*8+4), ymm08); 
}

inline void vectorized8x8(double * restrict A, double * restrict B, double * restrict C) {
    __assume_aligned(A, BYTE_ALIGN);
    __assume_aligned(B, BYTE_ALIGN);
    __assume_aligned(C, BYTE_ALIGN);
    
    // adapted from:
    //     https://software.intel.com/en-us/articles/benefits-of-intel-avx-for-small-matrices

    __m256d ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,// piecewise store A
            ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
            ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,// piecewise store B
            ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31;

    // Read in the 8 rows of Matrix B into zmm registers
    ymm16 = _mm256_load_pd((double *) (B + 0*8)); ymm17 = _mm256_load_pd((double *) (B + 0*8 + 4));// row 1
    ymm14 = _mm256_load_pd((double *) (B + 1*8)); ymm19 = _mm256_load_pd((double *) (B + 1*8 + 4));// row 2
    ymm20 = _mm256_load_pd((double *) (B + 2*8)); ymm21 = _mm256_load_pd((double *) (B + 2*8 + 4));// row 3
    ymm22 = _mm256_load_pd((double *) (B + 3*8)); ymm23 = _mm256_load_pd((double *) (B + 3*8 + 4));// row 4
    ymm24 = _mm256_load_pd((double *) (B + 4*8)); ymm25 = _mm256_load_pd((double *) (B + 4*8 + 4));// row 5
    ymm26 = _mm256_load_pd((double *) (B + 5*8)); ymm27 = _mm256_load_pd((double *) (B + 5*8 + 4));// row 6
    ymm24 = _mm256_load_pd((double *) (B + 6*8)); ymm29 = _mm256_load_pd((double *) (B + 6*8 + 4));// row 7
    ymm30 = _mm256_load_pd((double *) (B + 7*8)); ymm31 = _mm256_load_pd((double *) (B + 7*8 + 4));// row 8

    // row by row computations
    row8x8(0, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(1, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(2, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(3, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(4, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(5, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(6, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
    row8x8(7, A, C,
           ymm00, ymm01, ymm02, ymm03, ymm04, ymm05, ymm06, ymm07,
           ymm08, ymm09, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
           ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23,
           ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31);
}

#include <stdio.h>
int main(int argc, char **argv) {
    /*
     * +-           -++-           -+   +-                      -+
     * | 1   2  3  4 || 17 18 19 20 |   |  250   260   270   280 |
     * | 5   6  7  8 || 21 22 23 24 | = |  618   644   670   696 |
     * | 9  10 11 12 || 25 26 27 28 |   |  986  1028  1070  1112 |
     * | 13 14 15 16 || 29 30 31 32 |   | 1354  1412  1470  1528 |
     * +-           -++-           -+   +-                      -+
     */
    // initialize aligned kernel memory
    A_KERNEL = (double *) _mm_malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double), BYTE_ALIGN);
    B_KERNEL = (double *) _mm_malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double), BYTE_ALIGN);
    C_KERNEL = (double *) _mm_malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double), BYTE_ALIGN);

    // add some dummy data
    int num = 1;
    printf("A_KERNEL: \n");
    for(int i = 0; i < KERNEL_SIZE; ++i) {
        for(int j = 0; j < KERNEL_SIZE; ++j) {
            A_KERNEL(i,j) = (double) num++;
            printf("%04f ", A_KERNEL(i,j));
        }
        printf("\n");
    }
    printf("\nB_KERNEL:\n");
    for(int i = 0; i < KERNEL_SIZE; ++i) {
        for(int j = 0; j < KERNEL_SIZE; ++j) {
            B_KERNEL(i,j) = (double) num++;
            printf("%04f ", B_KERNEL(i,j));
            C_KERNEL(i,j) = 0.0;
        }
        printf("\n");
    }

    vectorized8x8(A_KERNEL, B_KERNEL, C_KERNEL);

    printf("\nC_KERNEL:\n");
    for(int i = 0; i < KERNEL_SIZE; ++i) {
        for(int j = 0; j < KERNEL_SIZE; ++j) {
            printf("%04f ", C_KERNEL(i,j));
        }
        printf("\n");
    }

    printf("\n - - - - - - - - - \n");

    printf("+-           -++-           -+   +-                      -+\n");
    printf("| 1   2  3  4 || 17 18 19 20 |   |  250   260   270   280 |\n");
    printf("| 5   6  7  8 || 21 22 23 24 | = |  618   644   670   696 |\n");
    printf("| 9  10 11 12 || 25 26 27 28 |   |  986  1028  1070  1112 |\n");
    printf("| 13 14 15 16 || 29 30 31 32 |   | 1354  1412  1470  1528 |\n");
    printf("+-           -++-           -+   +-                      -+\n");

    _mm_free(A_KERNEL); A_KERNEL = NULL;
    _mm_free(B_KERNEL); B_KERNEL = NULL;
    _mm_free(C_KERNEL); C_KERNEL = NULL;

    return 0;
}
