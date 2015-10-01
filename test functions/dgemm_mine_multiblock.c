// INCOMPLETE

#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm with matrix copying";

#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 8)
#endif

#ifndef BIG_BLOCK_SIZE
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


void small_square_dgemm(const int lda, const double *restrict A, const double *restrict B, double *restrict C, const int coarse_i, const int coarse_j, const int coarse_k)
{
    const int n_small_blocks = BIG_BLOCK_SIZE/SMALL_BLOCK_SIZE; // Want to ensure that BIG_BLOCK_SIZE is a multiple of SMALL_BLOCK_SIZE

    const int num_block_M = (coarse_i + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_i) / SMALL_BLOCK_SIZE + ((lda - coarse_i)%SMALL_BLOCK_SIZE? 1 : 0));
    const int num_block_N = (coarse_j + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_j) / SMALL_BLOCK_SIZE + ((lda - coarse_j)%SMALL_BLOCK_SIZE? 1 : 0));
    const int num_block_K = (coarse_k + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_k) / SMALL_BLOCK_SIZE + ((lda - coarse_k)%SMALL_BLOCK_SIZE? 1 : 0));

    // Need to think about this
    int small_bi, small_bj, small_bk;
	int j, k, i;
    for (small_bj = 0; small_bj < num_block_N; ++small_bj) {
        //const int fine_j = small_bj * SMALL_BLOCK_SIZE;
        j = coarse_j + small_bj * SMALL_BLOCK_SIZE;
        for (small_bk = 0; small_bk < num_block_K; ++small_bk) {
            //const int fine_k = small_bk * SMALL_BLOCK_SIZE;
            k = coarse_k + small_bk * SMALL_BLOCK_SIZE;
            for (small_bi = 0; small_bi < num_block_M; ++small_bi) {
                //const int fine_i = small_bi * SMALL_BLOCK_SIZE;
                i = coarse_i + small_bi * SMALL_BLOCK_SIZE;
                do_block(lda, A, B, C, i, j, k);
            }
        }
    }
}


void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *restrict C)
{
    //const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
	const int n_big_blocks = M / BIG_BLOCK_SIZE + (M%BIG_BLOCK_SIZE? 1 : 0);

	double* Anew = (double*) malloc(M*M*sizeof(double));
    double* Bnew = (double*) malloc(M*M*sizeof(double));
	double* Cnew = (double*) malloc(M*M*sizeof(double));
	memset(Cnew, 0, M*M*sizeof(double));

	int blocki, blockj, irange, jrange, i, j;
	for (blockj = 0; blockj < n_blocks; ++blockj) {
		for (blocki = 0; blocki < n_blocks; ++blocki) {
			irange = ((blocki+1)*BIG_BLOCK_SIZE <= M? BIG_BLOCK_SIZE : M-(blocki*BIG_BLOCK_SIZE));
			jrange = ((blockj+1)*BIG_BLOCK_SIZE <= M? BIG_BLOCK_SIZE : M-(blockj*BIG_BLOCK_SIZE));
			for (j = 0; j < jrange; ++j) {
				for (i = 0; i < irange; ++i) {
					Anew[blockj*M*BIG_BLOCK_SIZE + blocki*BIG_BLOCK_SIZE*jrange + i + j*irange] = A[blockj*M*BIG_BLOCK_SIZE + j*M + blocki*BIG_BLOCK_SIZE +i];
					Bnew[blockj*M*BIG_BLOCK_SIZE + blocki*BIG_BLOCK_SIZE*jrange + i + j*irange] = B[blockj*M*BIG_BLOCK_SIZE + j*M + blocki*BIG_BLOCK_SIZE +i];
					//Cnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = C[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
				}
			}
		}
	}

	int big_bi, big_bj, big_bk;
    for (big_bj = 0; big_bj < n_big_blocks; ++big_bj) {
		const int coarse_j = big_bj * BIG_BLOCK_SIZE;
        for (big_bk = 0; big_bk < n_big_blocks; ++big_bk) {
            const int coarse_k = big_bk * BIG_BLOCK_SIZE;
            for (big_bi = 0; big_bi < n_big_blocks; ++big_bi) {
				const int coarse_i = big_bi * BIG_BLOCK_SIZE;
                //do_block(M, A, B, C, i, j, k);
                small_square_dgemm(M, Anew, Bnew, Cnew, coarse_i, coarse_j, coarse_k);
            }
        }
    }

	for (blockj = 0; blockj < n_blocks; ++blockj) {
		for (blocki = 0; blocki < n_blocks; ++blocki) {
			irange = ((blocki+1)*BIG_BLOCK_SIZE <= M? BIG_BLOCK_SIZE : M-(blocki*BIG_BLOCK_SIZE));
			jrange = ((blockj+1)*BIG_BLOCK_SIZE <= M? BIG_BLOCK_SIZE : M-(blockj*BIG_BLOCK_SIZE));
			for (j = 0; j < jrange; ++j) {
				for (i = 0; i < irange; ++i) {
					C[blockj*M*BIG_BLOCK_SIZE + j*M + blocki*BIG_BLOCK_SIZE +i] = Cnew[blockj*M*BIG_BLOCK_SIZE + blocki*BIG_BLOCK_SIZE*jrange + i + j*irange];
				}
			}
		}
	}
}