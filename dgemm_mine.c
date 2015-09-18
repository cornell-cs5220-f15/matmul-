const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 100)
#endif

//modified the dgemm_blocked code

/*
 A is M-by-K
 B is K-by-N
 C is M-by-N
 
 lda is the leading dimension of the matrix (the M of square_dgemm).
*/

//double smallC[BLOCK_SIZE][BLOCK_SIZE], smallB[BLOCK_SIZE][BLOCK_SIZE], smallA[BLOCK_SIZE][BLOCK_SIZE];

void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
  int i, j, k;
  for (j = 0; j < N; ++j) {
    for (k = 0; k < K; ++k) {
      //            double cij = C[j*lda+i];
      for (i = 0; i < M; ++i) {
	*(C+j*lda+i) += *(A+k*lda+i) * *(B+j*lda+k);
      }
      //            *(C+j*lda+i) = cij;
    }
  }
}

void do_block(const int lda,
              const double *restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k)
{
  const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
  const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
  const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
  int ii, jj;

  //for (jj=0; jj<BLOCK_SIZE; ++jj){
    //for(ii=0;ii<BLOCK_SIZE;ii++) {
      // smallA[jj*BLOCK_SIZE+ii]= *(A+k*lda+i)
	// }
    // }
  // smallA = 
    // smallB = 
    //  smallC =   
  basic_dgemm(lda, M, N, K,
	      A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
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
