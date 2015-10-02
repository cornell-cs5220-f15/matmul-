//========================================================================
// dgemm_mine.c
//========================================================================
// DGEMM blocked implementation with AVX extensions, copy optimization,
// and alternate loop ordering. This is the culmination of the three
// primary optimizations explored for this assignment. In order to
// facilitate vector loads/stores for any matrix dimension, we must
// choose a block size that is evenly divisible by the SIMD width
// (256b = 4 doubles). In addition, we still want the blocks to be sized
// such that the memory footprint for every iteration fits entirely in
// the L1 cache.
//
// If the block size is set to be evenly divisible by the SIMD width, we
// can use copy optimization to ensure 256b-alignment of all columns
// across all blocks for any matrix dimension, obviating the need for the
// less efficient masked or unaligned vector loads/stores. We always
// enough space in the scratchpad to hold entire blocks for all three
// matrices so that for the corner cases when the remaining data does not
// fill up the entire block, we can still do vector loads/stores on trash
// values as long as we do not write these results back to the output
// matrix.

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

const char* dgemm_desc =
    "Blocked dgemm with AVX extensions, copy optimization, and alternate loop ordering.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
inline __attribute__((always_inline))
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *restrict A, const double *restrict B,
                 double *restrict C)
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
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            for (i = 0; i < num_wide_ops; ++i) {
                // Load partial products in result matrix. We can use a
                // vector load to efficiently load 4 doubles at once with a
                // unit-stride. If the block size is set to be evenly
                // divisible by the SIMD width, we can use copy optimization
                // to ensure all vector loads/stores are properly aligned.
                double       *cij_vec_addr = C + (j * lda) + (i * 4);
                __m256d       cij_vec      = _mm256_load_pd(cij_vec_addr);
                const double *aik_vec_addr = A + (k * lda) + (i * 4);
                __m256d       aik_vec      = _mm256_load_pd(aik_vec_addr);
                double        bkj          = B[j*lda+k];
                __m256d       bkj_vec      = _mm256_set1_pd(bkj);

                cij_vec = _mm256_fmadd_pd(aik_vec, bkj_vec, cij_vec);

                // Store partial products back into result matrix
                _mm256_store_pd(cij_vec_addr, cij_vec);
            }
        }
    }
}

inline __attribute__((always_inline))
void do_block(const int lda,
              const double *restrict A, const double *restrict B,
              double *restrict C, double *restrict D,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    // Take the blocks we need for this iteration and copy the data into
    // a contiguous cache-line-aligned scratchpad in memory. Matrices
    // A/B/C are laid out consecutively in the same scratchpad.
    double *D_A = D;
    double *D_B = D + BLOCK_SIZE * BLOCK_SIZE;
    double *D_C = D + 2 * BLOCK_SIZE * BLOCK_SIZE;

    const double *A_block = A + i + k*lda;
    const double *B_block = B + k + j*lda;
          double *C_block = C + i + j*lda;

    // Stripmine columns from the blocks into the scratchpad. Note that
    // we need to increment by multiples of BLOCK_SIZE for the scratchpad
    // and multiples of lda for the input matrices every iteration.
    int idx;
    for (idx = 0; idx < K; ++idx)
      memcpy(D_A + idx*BLOCK_SIZE, A_block + idx*lda, 8 * M);
    if (i == 0)
      for (idx = 0; idx < N; ++idx)
        memcpy(D_B + idx*BLOCK_SIZE, B_block + idx*lda, 8 * K);
    for (idx = 0; idx < N; ++idx)
      memcpy(D_C + idx*BLOCK_SIZE, C_block + idx*lda, 8 * M);

    // Call the computation kernel. The lda argument must be set to
    // BLOCK_SIZE to reflect how the blocks are laid out in the
    // scratchpad.
    basic_dgemm(BLOCK_SIZE, M, N, K, D_A, D_B, D_C);

    // Copy the results back to the result matrix
    for (idx = 0; idx < N; ++idx)
      memcpy(C_block + idx*lda, D_C + idx*BLOCK_SIZE, 8 * M);
}

void square_dgemm(const int M, const double *restrict A, const double *restrict B,
                  double *restrict C)
{
    // Scratchpad for copying blocks into cache-line-aligned (64B)
    // memory. We allocate enough memory for three identically sized
    // blocks.
    double D[3*BLOCK_SIZE*BLOCK_SIZE] __attribute__((aligned(64)));
//    posix_memalign((void**)&D, 64, 3 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // Divide and conquer computation into blocks
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                do_block(M, A, B, C, D, i, j, k);
            }
        }
    }
}

