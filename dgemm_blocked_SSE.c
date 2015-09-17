#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

const char* dgemm_desc = "SSE Blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    double b_copy;
    __m256d ymm_a, ymm_b, ymm_c;
    int i, j, k;
    printf("Calling basic.\n");

    //ADD LDA DEPENDENCY!!
    if(M == 4 && N == 4 && K == 4) {
        for(i = 0; i<4; i++) {
            ymm_c = _mm256_load_pd(C + i*lda);

            ymm_a = _mm256_load_pd(A);
            b_copy = B[i * lda];
            ymm_b = _mm256_broadcast_sd(&b_copy);
            ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

            ymm_a = _mm256_load_pd(A + 1*lda);
            b_copy = B[i * lda + 1];
            ymm_b = _mm256_broadcast_sd(&b_copy);
            ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

            ymm_a = _mm256_load_pd(A + 2*lda);
            b_copy = B[i * lda + 2];
            ymm_b = _mm256_broadcast_sd(&b_copy);
            ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

            ymm_a = _mm256_load_pd(A + 3*lda);
            b_copy = B[i * lda + 3];
            ymm_b = _mm256_broadcast_sd(&b_copy);
            ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

            _mm256_store_pd(C + i * lda, ymm_c);
        }
    } else {
        for (i = 0; i < M; ++i) {
            for (j = 0; j < M; ++j) {
                double cij = C[j*M+i];
                for (k = 0; k < M; ++k)
                    cij += A[k*M+i] * B[j*M+k];
                C[j*M+i] = cij;
            }
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    printf("Called block\n");
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    printf("Called dgemm\n");
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                /* do_block(M, A, B, C, i, j, k); */
                do_block(M, A, B, C, 0, 0, 0);
            }
        }
    }
}

