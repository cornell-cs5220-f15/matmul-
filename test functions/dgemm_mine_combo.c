#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm with matrix copying";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif
//
/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
/////////////////////////////WITHOUT COPY//////////////////////
// WITHOUT COPY OPT
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *restrict A, const double *restrict B, double *restrict C)
{
    int i, j, k;
		for (j = 0; j < N; ++j) { 
            for (k = 0; k < K; ++k) {
				/*for (i = 0; (i+3) < M; i+=4) {
					C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
					C[j*lda+(i+1)] += A[k*lda+(i+1)] * B[j*lda+k];
					C[j*lda+(i+2)] += A[k*lda+(i+2)] * B[j*lda+k];
					C[j*lda+(i+3)] += A[k*lda+(i+3)] * B[j*lda+k];
				}
				for (; i < M; ++i){
					C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
				}*/
				for (i = 0; i < M; ++i){
					C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
				}
			}
		}
}

void do_block(const int lda,
              const double *restrict A, const double *restrict B, double *restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}



////////////////////////////WITH COPY//////////////////////////
void basic_dgemm_cpy(const int lda, const int M, const int N, const int K,
                 const double *restrict Anew, const double *restrict Bnew, double *C)
{
    int i, j, k;
		for (j = 0; j < N; ++j) {
			for (k = 0; k < K; ++k) { 
				for (i = 0; i < M; ++i) {
					C[j*lda+i] += Anew[k*M+i] * Bnew[j*K+k];
				}
			}
		}
}

void do_block_cpy(const int lda,
              const double *restrict Anew, const double *restrict Bnew, double *restrict C,
              const int bi, const int bj, const int bk)
{
    const int M = ((bi+1)*BLOCK_SIZE > lda? lda-(bi*BLOCK_SIZE) : BLOCK_SIZE);
    const int N = ((bj+1)*BLOCK_SIZE > lda? lda-(bj*BLOCK_SIZE) : BLOCK_SIZE);
    const int K = ((bk+1)*BLOCK_SIZE > lda? lda-(bk*BLOCK_SIZE) : BLOCK_SIZE);
	
	basic_dgemm_cpy(lda, M, N, K,
				Anew + bk*lda*BLOCK_SIZE + bi*BLOCK_SIZE*K,
				Bnew + bj*lda*BLOCK_SIZE + bk*BLOCK_SIZE*N,
				C + bj*lda*BLOCK_SIZE + bi*BLOCK_SIZE);
  }

void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *restrict C)
{
	if (M > 200){

		const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

		double* Anew = (double*) malloc(M*M*sizeof(double));
		double* Bnew = (double*) malloc(M*M*sizeof(double));

		int blocki, blockj, irange, jrange, i, j;
		for (blockj = 0; blockj < n_blocks; ++blockj) {
			for (blocki = 0; blocki < n_blocks; ++blocki) {
				irange = ((blocki+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blocki*BLOCK_SIZE));
				jrange = ((blockj+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-(blockj*BLOCK_SIZE));
				for (j = 0; j < jrange; ++j) {
					for (i = 0; i < irange; ++i) {
						Anew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = A[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
						Bnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = B[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
					}
				}
			}
		}


		int bi, bj, bk;
		for (bj = 0; bj < n_blocks; ++bj) {
			//const int j = bj * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks; ++bk) {
				//const int k = bk * BLOCK_SIZE;
				for (bi = 0; bi < n_blocks; ++bi) {
					//const int i = bi * BLOCK_SIZE;
					do_block_cpy(M, Anew, Bnew, C, bi, bj, bk);
				}
			}
		}
	}
	else{

		const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
		int bi, bj, bk;

		for (bj = 0; bj < n_blocks; ++bj) {
		const int j = bj * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks; ++bk) {
			const int k = bk * BLOCK_SIZE;
				for (bi = 0; bi < n_blocks; ++bi) {
					const int i = bi * BLOCK_SIZE;
					do_block(M, A, B, C, i, j, k);
				}
			}
		}
	}
}
