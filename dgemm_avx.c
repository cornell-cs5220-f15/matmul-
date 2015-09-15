//========================================================================
// dgemm_blocked_sse.c
//========================================================================
// DGEMM blocked implementation with SSE extensions.

#include <immintrin.h>

const char* dgemm_desc = "Blocked dgemm with SSE extensions.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

//// Need to set the strides for gathering elements of a row into SIMD
//// registers. Assumes column-major layout for matrices.
//__m256d vec_strides = _mm256_set_pd((double)lda*0, (double)lda*1,
//                                    (double)lda*2, (double)lda*3);

// Helper for moving gather data from memory into SIMD register
__m256d gather_vec(const int lda, const double* addr) {
  double d0 = addr[lda*0];
  double d1 = addr[lda*1];
  double d2 = addr[lda*2];
  double d3 = addr[lda*3];
  return _mm256_set_pd(d3, d2, d1, d0);
}

// Helper for storing scatter data in SIMD register to memory
void scatter_vec(const int lda, double* addr, __m256d data) {
  // Extract lower 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d lower_data = _mm256_extractf128_pd(data, 0);
  _mm_storel_pd(addr+lda*0, lower_data);
  _mm_storeh_pd(addr+lda*1, lower_data);

  // Extract higher 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d higher_data = _mm256_extractf128_pd(data, 1);
  _mm_storel_pd(addr+lda*2, higher_data);
  _mm_storeh_pd(addr+lda*3, higher_data);
}

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    // Main computation loop for current block
    int i, j, k;
    for (i = 0; i < M; ++i) {

        // Use SIMD extensions to parallelize computation across multiple
        // columns in matrix B with the same row in matrix A. The number
        // of SIMD operations required for all columns is the number of
        // elements (doubles) in a row divided by the number of doubles
        // supported by a SIMD operation (256b -> 4 doubles).

        int num_wide_ops = (N + 3) / 4; // ceil(N/4)

        for (j = 0; j < num_wide_ops; ++j) {

            // Load partial products in result matrix. Need to use
            // gathers here instead of loads because of the column-major
            // layout for matrices.
            double *cij_vec_addr = C + (j * 4 * lda) + i;
            __m256d cij_vec      = gather_vec(lda, cij_vec_addr);

            // Accumulate fused multiply-adds for the current column
            // group across the block length.
            for (k = 0; k < K; ++k) {
                double        aik          = A[k*lda+i];
                __m256d       aik_vec      = _mm256_set1_pd(aik);
                const double *bkj_vec_addr = B + (j * 4 * lda) + k;
                __m256d       bkj_vec      = gather_vec(lda, bkj_vec_addr);
                cij_vec = _mm256_fmadd_pd(aik_vec, bkj_vec, cij_vec);
            }

            // Store partial products back into result matrix
            scatter_vec(lda, cij_vec_addr, cij_vec);
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
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

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

