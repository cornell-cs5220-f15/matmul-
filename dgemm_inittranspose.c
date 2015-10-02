#include <stdlib.h>

const char* dgemm_desc = "Our optimized dgemm.";

/* Global note: This code assumes the matrices are indexed in COLUMN-MAJOR order.
 * "i" variables increment the column of a pointer. "j" variables increment the
 * row and are multiplied by lda (the original M dimension of the matrix) in
 * order to do so. "k" is both a row and a column of the "inner dimension" and
 * depends on whether it's being multiplied by lda (in which case it's a row).
 */

/* Three 36x36 matrices will fit in the 32KB L1 cache.
 * This can be increased to 44 if we can find a way to not cache matrix C.*/
#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 64)
#endif
/* Three 103x103 matrices will fit in the 256KB L2 cache.
 * This can be increased to 126 if we can find a way to not cache matrix C.*/
#ifndef LARGE_BLOCK_SIZE
#define LARGE_BLOCK_SIZE ((int) 256)
#endif

#define MIN(x,y) (x > y ? y : x)
/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
  This function now expects A to be laid out in row-major order, while B and C
  are still column-major.
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
			double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k+i*lda] * B[j*lda+k];
            }
			C[j*lda+i] = cij;
        }
    }
}

/* Multiplies two small blocks of A and B that start at the offsets given by i,
 * j, and k. They should be squares of size SMALL_BLOCK_SIZE, but if any 
 * dimension exceeds the size of the containing large block (given by M, N, and
 * K), the blocks are reduced to rectangles of size M-by-K, K-by-N, and M-by-N.
 *
 * Note that A is in row-major order, while B and C are column-major.
 */
void do_small_block(const int lda, const int M, const int N, const int K,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int small_i_offset, const int small_j_offset, const int small_k_offset)
{
    const int SMALL_M = (small_i_offset+SMALL_BLOCK_SIZE > M? M-small_i_offset : SMALL_BLOCK_SIZE);
    const int SMALL_N = (small_j_offset+SMALL_BLOCK_SIZE > N? N-small_j_offset : SMALL_BLOCK_SIZE);
    const int SMALL_K = (small_k_offset+SMALL_BLOCK_SIZE > K? K-small_k_offset : SMALL_BLOCK_SIZE);
    basic_dgemm(lda, SMALL_M, SMALL_N, SMALL_K,
                A + small_i_offset*lda + small_k_offset, 
				B + small_k_offset + small_j_offset*lda,
				C + small_i_offset + small_j_offset*lda);
}

/* Multiplies two large blocks of A and B that start at the offsets given by i,
 * j, and k, by further splitting them into smaller blocks and calling do_small_block.
 * The blocks should be squares of LARGE_BLOCK_SIZE, but their sizes are 
 * calculated as specified above in case of an irregular size block at the edge
 * of the matrix: A is M-by-K, B is K-by-N, C is M-by-N. 
 *
 * Note that A is in row-major order, while B and C are column-major. Thus, the 
 * i offset is treated differently for A and C although it represents a column
 * increment in both cases.
 */
void do_large_block(const int lda,
		const double* restrict A, const double* restrict B, double * restrict C,
		const int large_i_offset, const int large_j_offset, const int large_k_offset)
{
    const int M = (large_i_offset+LARGE_BLOCK_SIZE > lda? lda-large_i_offset : LARGE_BLOCK_SIZE);
    const int N = (large_j_offset+LARGE_BLOCK_SIZE > lda? lda-large_j_offset : LARGE_BLOCK_SIZE);
    const int K = (large_k_offset+LARGE_BLOCK_SIZE > lda? lda-large_k_offset : LARGE_BLOCK_SIZE);
    const int n_blocks_i = M / SMALL_BLOCK_SIZE + (M % SMALL_BLOCK_SIZE? 1 : 0);
    const int n_blocks_j = N / SMALL_BLOCK_SIZE + (N % SMALL_BLOCK_SIZE? 1 : 0);
    const int n_blocks_k = K / SMALL_BLOCK_SIZE + (K % SMALL_BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks_i; ++bi) {
        const int small_i_offset = bi * SMALL_BLOCK_SIZE;
        for (bj = 0; bj < n_blocks_j; ++bj) {
            const int small_j_offset = bj * SMALL_BLOCK_SIZE;
            for (bk = 0; bk < n_blocks_k; ++bk) {
                const int small_k_offset = bk * SMALL_BLOCK_SIZE;
                do_small_block(lda, M, N, K,
				A + large_i_offset*lda + large_k_offset, 
				B + large_k_offset + large_j_offset*lda, 
				C + large_i_offset + large_j_offset*lda, 
				small_i_offset, small_j_offset, small_k_offset);
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C) 
{
    const int n_blocks = M / LARGE_BLOCK_SIZE + (M%LARGE_BLOCK_SIZE? 1 : 0);
    //Transpose matrix A to row-major order, since it is traversed in row-major order
    double* At = malloc(M*M*sizeof(double));
    int i, j;
    for(i = 0; i < M; ++i) {
		for(int j=0; j < M; ++j) {
			At[i + j*M] = A[i*M + j];
		}
    }

    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int large_i_offset = bi * LARGE_BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int large_j_offset = bj * LARGE_BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int large_k_offset = bk * LARGE_BLOCK_SIZE;
                do_large_block(M, At, B, C, large_i_offset, large_j_offset, large_k_offset);
            }
        }
    }
	free(At);
}

