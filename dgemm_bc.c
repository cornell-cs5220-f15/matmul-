const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *a, const double *b, double *C)
{
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[j*lda+i];
      for (k = 0; k < K; ++k) {
	cij += a[k*M+i] * b[j*K+k];
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
      
  double a[M * K];
  double b[K * N];

  for (int t = 0; t < M; ++t) {
    for (int s = 0; s < K; ++s) {
      a[t + s * M] = A[i + t + (k + s) * lda];
    } 
  }

  for (int t = 0; t < K; ++t) {
    for (int s = 0; s < N; ++s) {
      b[t + s * K] = B[k + t + (j + s) * lda];
    } 
  }

  basic_dgemm(lda, M, N, K,
	      a, b, C + i + j*lda);
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

