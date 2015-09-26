#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "256 copy optimized blocks with transposition, restrict keyword and xCORE-AVX2 flag";

#ifndef BLOCK2_SIZE
#define BLOCK2_SIZE ((int) 128)
#endif

#ifndef BLOCK3_SIZE
#define BLOCK3_SIZE ((int) 256)
#endif

double temp[4] __attribute__((aligned(64)));

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void inner_kernel(const int lda,
    const int M, const int N, const int K, double * restrict C, const double * restrict bufA, const double * restrict bufB)
{
    int i, j, k;
    __assume_aligned(bufA, 64);
    __assume_aligned(bufB, 64);
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            // double cij = C[j*lda+i];
            // __m256d ymm2 = _mm256_setzero_pd();

            // for (k = 0; k < K/4*4; k+=4) {

            //     __m256d ymm0 = _mm256_load_pd(bufA+i*BLOCK3_SIZE+k);
            //     __m256d ymm1 = _mm256_load_pd(bufB+j*BLOCK3_SIZE+k);
            //     __m256d ymm3 = _mm256_mul_pd(ymm0, ymm1);
            //     ymm2 = _mm256_add_pd(ymm2, ymm3);
            //     // cij += bufA3[i*BLOCK3_SIZE+k] * bufB3[j*BLOCK3_SIZE+k];
            //     // cij += bufA3[i*BLOCK3_SIZE+k+1] * bufB3[j*BLOCK3_SIZE+k+1];
            //     // cij += bufA3[i*BLOCK3_SIZE+k+2] * bufB3[j*BLOCK3_SIZE+k+2];
            //     // cij += bufA3[i*BLOCK3_SIZE+k+3] * bufB3[j*BLOCK3_SIZE+k+3];
            // }
            // for (k = K/4*4; k < K; ++k) {
            //     cij += bufA[i*BLOCK3_SIZE+k] * bufB[j*BLOCK3_SIZE+k];
            // }

            // _mm256_store_pd(temp, ymm2);
            // cij += temp[0] + temp[1] + temp[2] + temp[3];

            for (i = 0; i < M; ++i) {
                C[j*lda+i] += bufA[i*BLOCK3_SIZE+k] * bufB[j*BLOCK3_SIZE+k];
            }
            // C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k, double * restrict bufA, double * restrict bufB)
{
    const int M = (i+BLOCK3_SIZE > lda? lda-i : BLOCK3_SIZE);
    const int N = (j+BLOCK3_SIZE > lda? lda-j : BLOCK3_SIZE);
    const int K = (k+BLOCK3_SIZE > lda? lda-k : BLOCK3_SIZE);
    int a, b;
    __assume_aligned(bufB, 64);
    for (a = 0; a < N; ++a) {
        for (b = 0; b < K; ++b) {
            bufB[a*BLOCK3_SIZE+b] = B[(a+j)*lda+b+k];
        }
    }

    __assume_aligned(bufA, 64);
    for (a = 0; a < M; ++a) {
        for (b = 0; b < K; ++b) {
            bufA[a*BLOCK3_SIZE+b] = A[(b+k)*lda+a+i];
        }
    }
    inner_kernel(lda, M, N, K, C + i + j*lda, bufA, bufB);
}

void do_block2(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k, double * restrict bufA, double * restrict bufB)
{
    const int n_blocks = BLOCK2_SIZE / BLOCK3_SIZE + (BLOCK2_SIZE%BLOCK3_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int r = bi * BLOCK3_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int s = bj * BLOCK3_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int t = bk * BLOCK3_SIZE;
                do_block(lda, A, B, C, i+r, j+s, k+t, bufA, bufB);
            }
        }
    }
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    double *bufA = (double*) _mm_malloc(BLOCK3_SIZE*BLOCK3_SIZE*sizeof(double), 64);
    double *bufB = (double*) _mm_malloc(BLOCK3_SIZE*BLOCK3_SIZE*sizeof(double), 64);
    const int n_blocks = M / BLOCK3_SIZE + (M%BLOCK3_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK3_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK3_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK3_SIZE;
                do_block(M, A, B, C, i, j, k, bufA, bufB);
            }
        }
    }

    _mm_free(bufA);
    _mm_free(bufB);
}


