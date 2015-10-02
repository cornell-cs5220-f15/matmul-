#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm with matrix copying";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *restrict Anew, const double *restrict Bnew, double *Cnew)
{
    int i, j, k;
		for (j = 0; j < N; ++j) {
			for (k = 0; k < K; ++k) { 
				for (i = 0; i < M; ++i) {
					Cnew[j*M+i] += Anew[k*M+i] * Bnew[j*K+k];
				}
			}
		}
}

void do_block(const int lda,
              const double *restrict Anew, const double *restrict Bnew, double *restrict Cnew,
              const int bi, const int bj, const int bk)
{
    const int M = ((bi+1)*BLOCK_SIZE > lda? lda-(bi*BLOCK_SIZE) : BLOCK_SIZE);
    const int N = ((bj+1)*BLOCK_SIZE > lda? lda-(bj*BLOCK_SIZE) : BLOCK_SIZE);
    const int K = ((bk+1)*BLOCK_SIZE > lda? lda-(bk*BLOCK_SIZE) : BLOCK_SIZE);
	
	basic_dgemm(lda, M, N, K,
				Anew + bk*lda*BLOCK_SIZE + bi*BLOCK_SIZE*K,
				Bnew + bj*lda*BLOCK_SIZE + bk*BLOCK_SIZE*N,
				Cnew + bj*lda*BLOCK_SIZE + bi*BLOCK_SIZE*N);
  }

void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

	double* Anew = (double*) malloc(M*M*sizeof(double));
    double* Bnew = (double*) malloc(M*M*sizeof(double));
	double* Cnew = (double*) malloc(M*M*sizeof(double));
	// Zero out Cnew
	memset(Cnew, 0, M*M*sizeof(double));

	int blocki, blockj, irange, jrange, i, j;
	for (blockj = 0; blockj < n_blocks; ++blockj) {
		for (blocki = 0; blocki < n_blocks; ++blocki) {
			irange = ((blocki+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blocki*BLOCK_SIZE));
			jrange = ((blockj+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blockj*BLOCK_SIZE));
			for (j = 0; j < jrange; ++j) {
				for (i = 0; i < irange; ++i) {
					Anew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = A[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
					Bnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = B[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
					//Cnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = C[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
				}
			}
		}
	}


    int bi, bj, bk;
	for (bj = 0; bj < n_blocks; ++bj) {
		for (bk = 0; bk < n_blocks; ++bk) {
			for (bi = 0; bi < n_blocks; ++bi) {
                do_block(M, Anew, Bnew, Cnew, bi, bj, bk);
            }
        }
    }

	for (blockj = 0; blockj < n_blocks; ++blockj) {
		for (blocki = 0; blocki < n_blocks; ++blocki) {
			irange = ((blocki+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blocki*BLOCK_SIZE));
			jrange = ((blockj+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blockj*BLOCK_SIZE));
			for (j = 0; j < jrange; ++j) {
				for (i = 0; i < irange; ++i) {
					C[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i] = Cnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange];
				}
			}
		}
	}
}