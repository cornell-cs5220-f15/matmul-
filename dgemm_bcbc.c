const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE1
#define BLOCK_SIZE1 ((int) 128)
#endif

#ifndef BLOCK_SIZE2
#define BLOCK_SIZE2 ((int) 16)
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
	cij += A[k*BLOCK_SIZE2+i] * B[j*BLOCK_SIZE2+k];
      }
      C[j*lda+i] = cij;
    }
  }
}

void do_block2(const int lda,
               const double *A, const double *B, double *C,
               const int i, const int j, const int k)
{
  const int M = (i+BLOCK_SIZE2 > BLOCK_SIZE1? BLOCK_SIZE1-i : BLOCK_SIZE2);
  const int N = (j+BLOCK_SIZE2 > BLOCK_SIZE1? BLOCK_SIZE1-j : BLOCK_SIZE2);
  const int K = (k+BLOCK_SIZE2 > BLOCK_SIZE1? BLOCK_SIZE1-k : BLOCK_SIZE2);
      
  double AA[BLOCK_SIZE2 * BLOCK_SIZE2] = {0};
  double BB[BLOCK_SIZE2 * BLOCK_SIZE2] = {0};

  for (int t = 0; t < M; ++t) {
    for (int s = 0; s < K; ++s) {
      AA[t + s * BLOCK_SIZE2] = A[i + t + (k + s) * BLOCK_SIZE1];
    } 
  }

  for (int t = 0; t < K; ++t) {
    for (int s = 0; s < N; ++s) {
      BB[t + s * BLOCK_SIZE2] = B[k + t + (j + s) * BLOCK_SIZE1];
    } 
  }

  basic_dgemm(lda, M, N, K,
	      AA, BB, C + i + j * lda);
}

void do_block1(const int lda,
               const double *A, const double *B, double *C,
               const int i, const int j, const int k)
{
  const int M = (i+BLOCK_SIZE1 > lda? lda-i : BLOCK_SIZE1);
  const int N = (j+BLOCK_SIZE1 > lda? lda-j : BLOCK_SIZE1);
  const int K = (k+BLOCK_SIZE1 > lda? lda-k : BLOCK_SIZE1);
      
  double AA[BLOCK_SIZE1 * BLOCK_SIZE1] = {0};
  double BB[BLOCK_SIZE1 * BLOCK_SIZE1] = {0};

  for (int t = 0; t < M; ++t) {
    for (int s = 0; s < K; ++s) {
      AA[t + s * BLOCK_SIZE1] = A[i + t + (k + s) * lda];
    } 
  }

  for (int t = 0; t < K; ++t) {
    for (int s = 0; s < N; ++s) {
      BB[t + s * BLOCK_SIZE1] = B[k + t + (j + s) * lda];
    } 
  }

  const int n_blocks = BLOCK_SIZE1 / BLOCK_SIZE2 + (BLOCK_SIZE1%BLOCK_SIZE2? 1 : 0);
  int bi, bj, bk;
  for (bi = 0; bi < n_blocks; ++bi) {
    const int ii = bi * BLOCK_SIZE2;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int jj = bj * BLOCK_SIZE2;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int kk = bk * BLOCK_SIZE2;
	if (ii + i < lda && jj + j < lda && kk + k < lda) 
	  do_block2(lda, AA, BB, C + i + j * lda, ii, jj, kk);
      }
    }
  }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  const int n_blocks = M / BLOCK_SIZE1 + (M%BLOCK_SIZE1? 1 : 0);
  int bi, bj, bk;
  for (bi = 0; bi < n_blocks; ++bi) {
    const int i = bi * BLOCK_SIZE1;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int j = bj * BLOCK_SIZE1;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int k = bk * BLOCK_SIZE1;
	do_block1(M, A, B, C, i, j, k);
      }
    }
  }
}

