#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm with matrix copying";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 112)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

////////////////////////// WITHOUT COPY /////////////////////////////
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *restrict A, const double *restrict B, double *restrict C)
{
    int i, j, k;
		for (j = 0; j < N; ++j) {
			for (k = 0; k < K; ++k) { 
				for (i = 0; i < M; ++i) {
					C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
				}
			}
		}
}

void do_block(const int lda,
              const double *restrict A, const double *restrict B, double *restrict C,
              const int i, const int j, const int k)
{
	// Determine dimensions of blocks [i, j], [i, k] and [k, j].
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    // Calculate block C[i, j] = block A[i, k] * block B[k, j].
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
  }

///////////////////////// WITH COPY /////////////////////////////////
void basic_dgemm_copy(const int lda, const int M, const int N, const int K,
                 const double *restrict Anew, const double *restrict Bnew, double *restrict Cnew)
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

void do_block_copy(const int lda,
              const double *restrict Anew, const double *restrict Bnew, double *restrict Cnew,
              const int bi, const int bj, const int bk)
{
	// Determine dimensions of blocks [bi, bj], [bi, bk] and [bk, bj].
    const int M = ((bi+1)*BLOCK_SIZE > lda? lda-(bi*BLOCK_SIZE) : BLOCK_SIZE);
    const int N = ((bj+1)*BLOCK_SIZE > lda? lda-(bj*BLOCK_SIZE) : BLOCK_SIZE);
    const int K = ((bk+1)*BLOCK_SIZE > lda? lda-(bk*BLOCK_SIZE) : BLOCK_SIZE);
	
	// Calculate block C[bi, bj] = block A[bi, bk] * block B[bk, bj].
	basic_dgemm_copy(lda, M, N, K,
				Anew + bk*lda*BLOCK_SIZE + bi*BLOCK_SIZE*K,
				Bnew + bj*lda*BLOCK_SIZE + bk*BLOCK_SIZE*N,
				Cnew + bj*lda*BLOCK_SIZE + bi*BLOCK_SIZE*N);
  }

///////////////////////// MAIN ////////////////////////////

void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *restrict C)
{
	/* If the size of the matrix is large, use copy optimization. Otherwise don't.
	We chose 200 as a cut-off point based on plots of the blocking method with and
	without copying. The cut-off point represents the point at which the overhead
	of copying A, B, and C is worthwhile.*/

    if (M > 200){
    	// Calculate the number of blocks along an edge of an M x M matrix.
	    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

	    // Create tempory M x M matrices Anew, Bnew, and Cnew. Set Cnew to zero.
		double* Anew = (double*) malloc(M*M*sizeof(double));
	    double* Bnew = (double*) malloc(M*M*sizeof(double));
		double* Cnew = (double*) malloc(M*M*sizeof(double));
		memset(Cnew, 0, M*M*sizeof(double));

		/* Realign the elements of A and B so that they are stored in continuous memory
		in Anew and Bnew. */
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

		// Iterate through all block multiplications to evaluate matrix product.
	    int bi, bj, bk;
		for (bj = 0; bj < n_blocks; ++bj) {
			for (bk = 0; bk < n_blocks; ++bk) {
				for (bi = 0; bi < n_blocks; ++bi) {
	                do_block_copy(M, Anew, Bnew, Cnew, bi, bj, bk);
	            }
	        }
	    }

	    // Re-copy the contents of Cnew to C for the final solution.
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
	else{
		// Calculate the number of blocks along an edge of an M x M matrix.
		const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
		int bi, bj, bk;

		// Iterate through all block multiplications to evaluate matrix product.
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