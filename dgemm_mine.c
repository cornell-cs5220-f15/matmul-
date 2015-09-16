const char* dgemm_desc = "My awesome dgemm.";

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

/* My attempt

void basic_dgemm(const int lda,
                const int M, const int N, const int K,
                const double *A, const double *B, double *C)
{
  // Assume that the matrix A is of M*K and B is of K*N, C is M*N
  // Also the matrix A should be in row-major and B,C are in column-major
  // The transpose should be done already.
  int i, j, k;
  for (j = 0; j < N; ++j){  // Since C is arranged column-wise, we don't stride
    for (i = 0; i < M; ++i){
      double cij = C[j*lda + i];
      for (k = 0; k <K; ++k){ // Recall that A is A.transpose()
        cij += A[k*lda + i] * B[j*lda + k];
      }
      C[j*M+i] = cij;
    }
  }
}

void do_block(const int lda,
             const double *A, const double *B, double *C,
             const int i, const int j, const int k){
  // We are only interested in square matrices.
  // We only worry about edge cases where the sub-matrices are not square.

  // submatrices are of sizes sA-M*K, sB-K*N, sC-M*N.
  // Because we want to extract sA from column-wise storage and then restore it row-wise.
  // We keep the outer loop i to be row of (A) and then the second loop j to be column of A
  // So i corresponds to K and j corresponds to M.
  // Lastly, k corresponds to N
  const int M = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
  const int N = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
  const int K = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);

  // Here it should obey that sC = sA * sB
  basic_dgemm(lda, M, N, K, A+j+i*lda, B+i+k*lda, C+j+k*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

  // basic_dgemm(M,M,M,A,B,C);
   const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1:0); // determine the # of blocks we need to complete the computation.
   int bi, bj, bk; // iterator for each block

   for (bi = 0; bi < n_blocks; ++bi){
     const int i = bi * BLOCK_SIZE; // block starting point for A
     for (bj = 0; bj < n_blocks; ++bj){
       const int j = bj * BLOCK_SIZE; // block starting point for B
       for (bk = 0; bk < n_blocks; ++bk){
         const int k = bk * BLOCK_SIZE; // block starting point for C
         do_block(M, A, B, C, i, j, k);
       }
     }
   }

}
*/
