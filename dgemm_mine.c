#include <stdlib.h>
#include <stdio.h>

const char* dgemm_desc = "Mine";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

void transpose_array(const int M, const double *A, double *copied)
{
	int row, column;

	for (column = 0; column < M; ++column){
		for (row = 0; row < M; ++row){
			copied[(row * M) + column] = A[(column * M) + row];
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
  int i, j, k;

  // write to C in row-major order
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[j*lda+i];

      // we can compute these before the inner loop so two less multiplication per cycle!
      int a_start = i*lda;
      int b_start = j*lda;

      for (k = 0; k < K; ++k) {
        cij += A[a_start + k] * B[b_start + k];
      }

      C[j*lda+i] = cij;
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
    double* a_transposed = (double*) malloc(M * M * sizeof(double));

    // create a transposed A in row-major order
    transpose_array(M, A, a_transposed);

    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, a_transposed, B, C, i, j, k);
            }
        }
    }

    free(a_transposed);
}
