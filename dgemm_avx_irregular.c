//========================================================================
// dgemm_avx_irregular.c
//========================================================================
// DGEMM blocked implementation with AVX extensions. Assumes a
// column-major data layout for matrices. Computation is parallelized
// to produce multiple elements in the same row of the output matrix. Due
// to the access pattern, we need to use manual gathers/scatters to
// load/store the vector data. AVX2 has separate intrinsics to do this,
// but the totient nodes do not support AVX2.

#include <immintrin.h>

const char* dgemm_desc = "Blocked dgemm with AVX extensions (gather/scatter).";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

// Helper for moving gather data from memory into SIMD register
__m256d gather_vec(const int lda, const double* addr, char mask) {
  double d0 = ((mask >> 0) & 0x1) ? addr[lda*0] : 0.0;
  double d1 = ((mask >> 1) & 0x1) ? addr[lda*1] : 0.0;
  double d2 = ((mask >> 1) & 0x1) ? addr[lda*2] : 0.0;
  double d3 = ((mask >> 1) & 0x1) ? addr[lda*3] : 0.0;
  return _mm256_set_pd(d3, d2, d1, d0);
}

// Helper for storing scatter data in SIMD register to memory
void scatter_vec(const int lda, double* addr, __m256d data, char mask) {
  // Extract lower 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d lower_data = _mm256_extractf128_pd(data, 0);
  if ((mask >> 0) & 0x1)
    _mm_storel_pd(addr+lda*0, lower_data);
  if ((mask >> 1) & 0x1)
    _mm_storeh_pd(addr+lda*1, lower_data);

  // Extract higher 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d higher_data = _mm256_extractf128_pd(data, 1);
  if ((mask >> 2) & 0x1)
    _mm_storel_pd(addr+lda*2, higher_data);
  if ((mask >> 3) & 0x1)
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
        // elements within a row in matrix B with the same element in
        // matrix A. Essentially parallelizes the outer loop for row
        // traversal in order to compute multiple elements in the same
        // row of the output matrix. The number of outer loop iterations
        // required for to compute all columns is the number of elements
        // (doubles) in a row divided by the number of doubles supported
        // by a SIMD operation (256b -> 4 doubles).

        int num_wide_ops = (N + 3) / 4; // ceil(N/4)

        for (j = 0; j < num_wide_ops; ++j) {

            // Calculate vector load/store mask to handle the last
            // iteration if the matrix dimensions are not evenly
            // divisible by the SIMD width.
            int  num_valid_ops = M % 4;
            int  is_last_iter = (i == num_wide_ops - 1);
            int  shamt        = (num_valid_ops && is_last_iter) ? 4 - num_valid_ops : 0;
            char mask_vec     = 0xf >> shamt;

            // Load partial products in result matrix. Need to use
            // gathers here instead of loads because of the column-major
            // layout for matrices.
            double *cij_vec_addr = C + (j * 4 * lda) + i;
            __m256d cij_vec      = gather_vec(lda, cij_vec_addr, mask_vec);

            // Accumulate fused multiply-adds for the current row group
            // across the block length.
            for (k = 0; k < K; ++k) {
                double        aik          = A[k*lda+i];
                __m256d       aik_vec      = _mm256_set1_pd(aik);
                const double *bkj_vec_addr = B + (j * 4 * lda) + k;
                __m256d       bkj_vec      = gather_vec(lda, bkj_vec_addr, mask_vec);
                cij_vec = _mm256_fmadd_pd(aik_vec, bkj_vec, cij_vec);
            }

            // Store partial products back into result matrix
            scatter_vec(lda, cij_vec_addr, cij_vec, mask_vec);
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

