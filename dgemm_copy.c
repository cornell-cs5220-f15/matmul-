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

const char* dgemm_desc = "Blocked dgemm with copy optimization (block_size=36).";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 36)
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
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C, double *D,
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
    for (idx = 0; idx < N; ++idx) {
      memcpy(D_B + idx*BLOCK_SIZE, B_block + idx*lda, 8 * K);
      memcpy(D_C + idx*BLOCK_SIZE, C_block + idx*lda, 8 * M);
    }

    // Call the computation kernel. The lda argument must be set to
    // BLOCK_SIZE to reflect how the blocks are laid out in the
    // scratchpad.
    basic_dgemm(BLOCK_SIZE, M, N, K, D_A, D_B, D_C);

    // Copy the results back to the result matrix
    for (idx = 0; idx < N; ++idx)
      memcpy(C_block + idx*lda, D_C + idx*BLOCK_SIZE, 8 * M);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Scratchpad for copying blocks into cache-line-aligned (64B)
    // memory. We allocate enough memory for three identically sized
    // blocks.
    double *D;
    posix_memalign((void**)&D, 64, 3 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // Divide and conquer computation into blocks
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, D, i, j, k);
            }
        }
    }
}

