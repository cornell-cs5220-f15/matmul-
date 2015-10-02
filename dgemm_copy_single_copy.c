//========================================================================
// dgemm_copy.c
//========================================================================
// DGEMM blocked implementation with copy optimization. In order to
// reduce conflict misses in the cache, we copy data from the blocks we
// need for each iteration into a contiguous cache-line-aligned
// scratchpad in memory. Ideally, we want the blocks to be sized such
// that the memory footprint for every iteration fits entirely in the L1
// cache. First-order calculations show that with a 32KB L1 cache, we are
// able to fit 4096 64b words without any conflicts, meaning we can fit
// three MxM blocks where M = floor( sqrt( 4096 / 3 ) ) = 36.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with copy optimization (block_size=36).";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

// Helper method for copying data from input matrices into scratchpad
// with the data from each block laid out in contiguous memory.
inline __attribute__((always_inline))
void load_copy(const int lda, const int n_blocks,
               double *A_copy, double *B_copy,
               const double *A, const double *B)
{
    const int edge_size = lda % BLOCK_SIZE;

    int i, j, k;
    for (j = 0; j < n_blocks; ++j) {
        const int is_j_edge = (j == n_blocks - 1);
        const int K = (is_j_edge && edge_size) ? edge_size : BLOCK_SIZE;

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
            A_copy += (BLOCK_SIZE - K) * BLOCK_SIZE;
            B_copy += (BLOCK_SIZE - K) * BLOCK_SIZE;
        }
    }
}

// Helper method for copying data from scratchpad into output matrix with
// the data reverted to the original memory layout.
inline __attribute__((always_inline))
void store_copy(const int lda, const int n_blocks,
                double *C, double *C_copy)
{
    const int edge_size = lda % BLOCK_SIZE;

    int i, j, k;
    for (j = 0; j < n_blocks; ++j) {
        const int is_j_edge = (j == n_blocks - 1);
        const int K = (is_j_edge && edge_size) ? edge_size : BLOCK_SIZE;

        for (i = 0; i < n_blocks; ++i) {
            const int is_i_edge = (i == n_blocks - 1);
            const int M = (is_i_edge && edge_size) ? edge_size : BLOCK_SIZE;

            int offset = (i + j * lda) * BLOCK_SIZE;
            double *C_block = C + offset;

            for (k = 0; k < K; ++k) {
                memcpy(C_block, C_copy, 8 * M);
                C_copy += BLOCK_SIZE; C_block += lda;
            }
            C_copy += (BLOCK_SIZE - K) * BLOCK_SIZE;
        }
    }
}

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
inline __attribute__((always_inline))
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k)
                cij += A[k*lda+i] * B[j*lda+k];
            C[j*lda+i] = cij;
        }
    }
}

inline __attribute__((always_inline))
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

    double *D = (double*) malloc(3 * copy_size * sizeof(double));
    double *A_copy = D;
    double *B_copy = D + copy_size;
    double *C_copy = D + 2 * copy_size;

    // Load data from input matrices into scratchpad. We re-arrange the
    // data once for both input matrices before any computation to
    // amortize the overhead of the copy. There is no need to copy the
    // result matrix before any partial products have been computed.
    load_copy(M, n_blocks, A_copy, B_copy, A, B);

    // Initialize result scratchpad to zero (past trials can contaminate)
    memset(C_copy, 0, copy_size * sizeof(double));

    // Divide and conquer computation into blocks
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, n_blocks, A_copy, B_copy, C_copy, i, j, k);
            }
        }
    }

    // Store data from scratchpad into result matrix
    store_copy(M, n_blocks, C, C_copy);

    // Free scratchpad
    free(D);
}

