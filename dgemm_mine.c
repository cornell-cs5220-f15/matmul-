<<<<<<< HEAD

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

// ended up not being beneficial to use copy optimization
/*
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
  double Bcopy;
  double Ccopy[M];   

  int i, j, k,c1;
    
  for (j = 0; j < N; ++j) {
    for (c1=0; c1<M; c1++){
      Ccopy[c1]=C[j*lda+c1];
    }    
    for (k = 0; k < K; ++k) {
      Bcopy=B[j*lda+k];
      for (i = 0; i < M; ++i) {
	double cij = Ccopy[i];
	cij += A[i*lda+k] * Bcopy;
        Ccopy[i] = cij;
      }
    }
    for (c1=0; c1<M; c1++){
      C[j*lda+c1]=Ccopy[c1];
    }
  }
  }
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, 
                 double* restrict C)
{
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[j*lda+i];
      for (k = 0; k < K; ++k) {
	cij += A[i*lda+k] * B[j*lda+k];
      }
      C[j*lda+i] = cij;
    }
  }
}


#include <stdlib.h>

const char* dgemm_desc = "My very awesome dgemm.";

void square_dgemm_L2(const int M, const double *A, const double *B, double *C, int i, int j, int k, int large_blk_size);
void square_dgemm_L1(const int M, const double *A, const double *B, double *C, int i, int j, int k, int large_blk_size);
void square_dgemm_regs(const int M, const double *A, const double *B, double *C, int i, int j, int k, int large_blk_size);

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256 //512//1024 
#endif

void make_transpose(const int M, const double* restrict A, double* restrict out)
{
<<<<<<< HEAD
  int i, j;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < M; ++j) {
      out[j*M + i] = A[i*M + j];
    }
  }
}

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N
  lda is the leading dimension of the matrix (the M of square_dgemm).
*/



void do_block(const int lda,
              const double* restrict A, const double* restrict B,
              double* restrict C, const int i, const int j, const int k, 
              const int my_blk_size)
{
  const int M = (i+my_blk_size > lda? lda-i : my_blk_size);
  const int N = (j+my_blk_size > lda? lda-j : my_blk_size);
  const int K = (k+my_blk_size > lda? lda-k : my_blk_size);
  basic_dgemm(lda, M, N, K,
	      A + k + i*lda, B + k + j*lda, C + i + j*lda);
}

// fits in L3 (2.5 MB)
void square_dgemm(const int M, const double* restrict A, 
                  const double* restrict B, double* restrict C)
{
  const int blk_size = 288;//320;
  const int n_blocks = M / blk_size + (M%blk_size? 1 : 0);
  int bi, bj, bk;
    
  double *A_T = (double*)malloc(M * M * sizeof(double));
  make_transpose(M, A, A_T);

  for (bi = 0; bi < n_blocks; ++bi) {
    const int i = bi * blk_size;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int j = bj * blk_size;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int k = bk * blk_size;
	square_dgemm_L2(M, A_T, B, C, i, j, k, blk_size);
      }
    }
  }
}

// fits in L2 (256 KB)
void square_dgemm_L2(const int M, const double* restrict A, 
                     const double* restrict B, double* restrict C, const int i, 
                     const int j, const int k, int large_blk_size)
{
  int my_blk_size = 96;//100;
  const int n_blocks = large_blk_size / my_blk_size + (large_blk_size%my_blk_size? 1 : 0);
  int bi, bj, bk;
  for (bi = 0; bi < n_blocks; ++bi) {
    const int ti = i + bi * my_blk_size;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int tj = j + bj * my_blk_size;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int tk = k + bk * my_blk_size;
	square_dgemm_L1(M, A, B, C, ti, tj, tk, my_blk_size);
      }
    }
  }
}

// fits in L1 (32 KB)
void square_dgemm_L1(const int M, const double* restrict A, 
                     const double* restrict B, double* restrict C, const int i,                      const int j, const int k, int large_blk_size)
{
  int my_blk_size = 32;//36;
  const int n_blocks = large_blk_size / my_blk_size + (large_blk_size%my_blk_size? 1 : 0);
  int bi, bj, bk;
  for (bi = 0; bi < n_blocks; ++bi) {
    const int ti = i + bi * my_blk_size;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int tj = j + bj * my_blk_size;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int tk = k + bk * my_blk_size;
	do_block(M, A, B, C, ti, tj, tk, my_blk_size);
	//square_dgemm_regs(M, A, B, C, ti, tj, tk, my_blk_size);
      }
    }
  }
}

// fits in regs
void square_dgemm_regs(const int M, const double* restrict A, 
                       const double* restrict B, double* restrict C, 
                       const int i, const int j, const int k, 
                       int large_blk_size)
{
  int my_blk_size = 8; //16; //23
  const int n_blocks = large_blk_size / my_blk_size + (large_blk_size%my_blk_size? 1 : 0);
  int bi, bj, bk;
  for (bi = 0; bi < n_blocks; ++bi) {
    const int ti = i + bi * my_blk_size;
    for (bj = 0; bj < n_blocks; ++bj) {
      const int tj = j + bj * my_blk_size;
      for (bk = 0; bk < n_blocks; ++bk) {
	const int tk = k + bk * my_blk_size;
	do_block(M, A, B, C, ti, tj, tk, my_blk_size);
      }
    }
  }
}




