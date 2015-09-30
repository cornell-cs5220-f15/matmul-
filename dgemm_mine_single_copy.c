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

// Helper method for copying data from input matrices into scratchpad
// with the data from each block laid out in contiguous memory.
void load_copy(const int lda, const int n_blocks,
               double *A_copy, double *B_copy,
               const double *A, const double *B)
{
    const int edge_size = lda % BLOCK_SIZE;

    int i, j, k;
    for (j = 0; j < n_blocks; ++j) {
        const int is_j_edge = (j == n_blocks - 1);
        const int K = (is_j_edge && edge_size) ? edge_size : BLOCK_SIZE;

        #pragma simd
        for (i = 0; i < n_blocks; ++i) {
            const int is_i_edge = (i == n_blocks - 1);
            const int M = (is_i_edge && edge_size) ? edge_size : BLOCK_SIZE;

            int offset = (i + j * lda) * BLOCK_SIZE;
            const double *A_block = A + offset;
            const double *B_block = B + offset;

            for (k = 0; k < K; ++k) {
                memcpy(A_copy, A_block, 8 * M);
                memcpy(B_copy, B_block, 8 * M);
                A_copy += BLOCK_SIZE; A_block += lda;
                B_copy += BLOCK_SIZE; B_block += lda;
            }
        }
    }
}

// Helper method for copying data from scratchpad into output matrix with
// the data reverted to the original memory layout.
void store_copy(const int lda, const int n_blocks,
                double *C, double *C_copy)
{
    const int edge_size = lda % BLOCK_SIZE;

    int i, j, k;
    for (j = 0; j < n_blocks; ++j) {
        const int is_j_edge = (j == n_blocks - 1);
        const int K = (is_j_edge && edge_size) ? edge_size : BLOCK_SIZE;

        #pragma simd
        for (i = 0; i < n_blocks; ++i) {
            const int is_i_edge = (i == n_blocks - 1);
            const int M = (is_i_edge && edge_size) ? edge_size : BLOCK_SIZE;

            int offset = (i + j * lda) * BLOCK_SIZE;
            double *C_block = C + offset;

            for (k = 0; k < K; ++k) {
                memcpy(C_block, C_copy, 8 * M);
                C_copy += BLOCK_SIZE; C_block += lda;
            }
        }
    }
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
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
//            #pragma simd vectorlength(4) linear(i:1)
//            #pragma prefetch A
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

void do_block(const int lda, const int n_blocks,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    // Data in each block is laid out in contiguous memory so the offset
    // to (i,j)th block is (i + j * n_blocks) * BLOCK_SIZE * BLOCK_SIZE.
    // The equation below assumes that the i/j/k arguments are already
    // multiplied by BLOCK_SIZE in order to handle the boundary
    // condition calculations above.
    const double *A_block = A + (i + k*n_blocks) * BLOCK_SIZE;
    const double *B_block = B + (k + j*n_blocks) * BLOCK_SIZE;
          double *C_block = C + (i + j*n_blocks) * BLOCK_SIZE;

    // Call the computation kernel. The lda argument must be set to
    // BLOCK_SIZE to reflect how the blocks are laid out in the
    // scratchpad.
    basic_dgemm(BLOCK_SIZE, M, N, K, A_block, B_block, C_block);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Scratchpad for copying blocks into cache-line-aligned (64B)
    // memory. Each matrix is re-arranged so that the data in a block is
    // in contiguous memory.

    const int n_blocks  = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    const int copy_size = n_blocks * n_blocks * BLOCK_SIZE * BLOCK_SIZE;

//    double *A_copy, *B_copy, *C_copy;
//    posix_memalign((void**)&A_copy, 64, copy_size * sizeof(double));
//    posix_memalign((void**)&B_copy, 64, copy_size * sizeof(double));
//    posix_memalign((void**)&C_copy, 64, copy_size * sizeof(double));
    double *D;
    posix_memalign((void**)&D, 64, 3 * copy_size * sizeof(double));
    double *A_copy = D;
    double *B_copy = D + copy_size;
    double *C_copy = D + 2 * copy_size;

    // Load data from input matrices into scratchpad. We re-arrange the
    // data once for both input matrices before any computation to
    // amortize the overhead of the copy. There is no need to copy the
    // result matrix before any partial products have been computed.
    load_copy(M, n_blocks, A_copy, B_copy, A, B);

    // Divide and conquer computation into blocks
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                do_block(M, n_blocks, A_copy, B_copy, C_copy, i, j, k);
            }
        }
    }

    // Store data from scratchpad into result matrix
    store_copy(M, n_blocks, C, C_copy);
}

