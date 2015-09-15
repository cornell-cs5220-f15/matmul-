//========================================================================
// dgemm_avx_regular.c
//========================================================================
// DGEMM blocked implementation with AVX extensions. Optimized for
// column-major data layout of matrices which allows us to use vector
// loads and stores. We need to ensure the matrices allocated in memory
// are aligned to doubles in order for this to work.

#include <immintrin.h>
#include <stdlib.h>
//#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with AVX extensions (vector load/store).";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

#define MASK_1 0xffffffffffffffff
#define MASK_0 0x0000000000000000

// Helper for calculating mask for vector loads/stores. Enabled mask
// values need to have the MSB set in order for the masked vector
// load/stores to detect it correctly so we use -1 instead of 1.
__m256i calculate_mask_vec(int num_valid_ops, int is_last_iter)
{
  // Lookup table implementation (more control flow, less computation)
  if (!is_last_iter)
    return _mm256_set_epi64x(MASK_1, MASK_1, MASK_1, MASK_1);
  else {
    switch (num_valid_ops) {
      case 0:
        return _mm256_set_epi64x(MASK_1, MASK_1, MASK_1, MASK_1);
      case 1:
        return _mm256_set_epi64x(MASK_0, MASK_0, MASK_0, MASK_1);
      case 2:
        return _mm256_set_epi64x(MASK_0, MASK_0, MASK_1, MASK_1);
      case 3:
        return _mm256_set_epi64x(MASK_0, MASK_1, MASK_1, MASK_1);
      default:
        exit(1); // Should not be here
    }
  }

//  // Bit masking implementation (less control flow, more computation)
//  char mask  = 0x0f;
//  int  shamt = (num_valid_ops && is_last_iter) ? 4 - num_valid_ops : 0;
//  mask >>= shamt;
//  return _mm256_set_epi64x(
//             ((mask >> 3) & 0x1) ? MASK_1 : MASK_0,
//             ((mask >> 2) & 0x1) ? MASK_1 : MASK_0,
//             ((mask >> 1) & 0x1) ? MASK_1 : MASK_0,
//             ((mask >> 0) & 0x1) ? MASK_1 : MASK_0
//         );
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
    // Use SIMD extensions to parallelize computation across multiple
    // elements within a column in matrix A with the same element in
    // matrix B. Essentially parallelizes the outer loop for column
    // traversal in order to compute multiple elements in the same column
    // of the output matrix. The number of outer loop iterations required
    // to compute all rows is the number of elements (doubles) in a
    // column divided by the number of doubles supported by a SIMD
    // operation (256b -> 4 doubles).
    int i, j, k;
    int num_wide_ops = (M + 3) / 4; // ceil(M/4)
    for (i = 0; i < num_wide_ops; ++i) {
        for (j = 0; j < N; ++j) {

            // Calculate vector load/store mask to handle the last
            // iteration if the matrix dimensions are not evenly
            // divisible by the SIMD width.
            int     is_last_iter = (i == num_wide_ops - 1);
            __m256i mask_vec     = calculate_mask_vec(M % 4, is_last_iter);

//            // DEBUG
//            printf("0: %d\n", _mm256_extract_epi64(mask_vec, 0));
//            printf("1: %d\n", _mm256_extract_epi64(mask_vec, 1));
//            printf("2: %d\n", _mm256_extract_epi64(mask_vec, 2));
//            printf("3: %d\n", _mm256_extract_epi64(mask_vec, 3));

            // Load partial products in result matrix. We can use a
            // vector load to efficiently load 4 doubles at once with a
            // unit-stride as long as we can ensure 64b-alignment.
            double *cij_vec_addr = C + (j * lda) + (i * 4);
            __m256d cij_vec      = _mm256_maskload_pd(
                                       cij_vec_addr, mask_vec);

            // Accumulate fused multiply-adds for the current column
            // group across the block length.
            for (k = 0; k < K; ++k) {
                const double *aik_vec_addr = A + (k * lda) + (i * 4);
                __m256d       aik_vec      = _mm256_maskload_pd(
                                                 aik_vec_addr, mask_vec);
                double        bkj          = B[j*lda+k];
                __m256d       bkj_vec      = _mm256_set1_pd(bkj);
                cij_vec = _mm256_fmadd_pd(aik_vec, bkj_vec, cij_vec);

//                // DEBUG
//                double tmp = 0.0;
//                __m128d lower = _mm256_extractf128_pd(aik_vec, 0);
//                _mm_storel_pd(&tmp, lower);
//                printf("result %d,%d,%d: %f\n", i, j, k, tmp);
            }

            // Store partial products back into result matrix
            _mm256_maskstore_pd(cij_vec_addr, mask_vec, cij_vec);
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

