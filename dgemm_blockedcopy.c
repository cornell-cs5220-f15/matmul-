const char* dgemm_desc = "Simple blocked dgemm. with matrix copying";

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
                 const double *Anew, const double *Bnew, double *C)
{
    int i, j, k;
		for (k = 0; k < K; ++k) {
			for (j = 0; j < N; ++j) { 
				for (i = 0; i < M; ++i) {
					C[j*lda+i] += Anew[k*M+i] * B[j*K+k];
					//C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
					// FIX THIS
            }
        }
    }
}

void do_block(const int lda,
              const double *Anew, const double *Bnew, double *C,
              const int bi, const int bj, const int bk)
{
    const int M = ((bi+1)*BLOCK_SIZE > lda? lda-(bi*BLOCK_SIZE) : BLOCK_SIZE);
    const int N = ((bj+1)*BLOCK_SIZE > lda? lda-(bj*BLOCK_SIZE) : BLOCK_SIZE);
    const int K = ((bk+1)*BLOCK_SIZE > lda? lda-(bk*BLOCK_SIZE) : BLOCK_SIZE);
	
	basic_dgemm(lda, M, N, K,
				Anew + bk*lda*BLOCK_SIZE + bi*BLOCK_SIZE*K,
				Bnew + bj*lda*BLOCK_SIZE + bk*BLOCK_SIZE*N,
				Cnew + bj*lda*BLOCK_SIZE + bi*BLOCK_SIZE)
                //A + i + k*lda, B + k + j*lda, C + i + j*lda)
				// FIX THIS
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

	double* Anew = (double*) malloc(M*M);
	double* Bnew = (double*) malloc(M*M);

	int blocki, blockj, irange, jrange, i, j;
	for (blockj = 0; blockj < n_blocks; ++blockj) {
		for (blocki = 0; blocki < n_blocks; ++blocki) {
			irange = ((blocki+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-blocki*BLOCK_SIZE);
			jrange = ((blockj+1)*BLOCK_SIZE <= M? BLOCK_SIZE : M-blockj*BLOCK_SIZE);
			for (j = 0; j < jrange; ++j) {
				for (i = 0; i < irange; ++i) {
					Anew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = A[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
					Bnew[blockj*M*BLOCK_SIZE + blocki*BLOCK_SIZE*jrange + i + j*irange] = B[blockj*M*BLOCK_SIZE + j*M + blocki*BLOCK_SIZE +i];
				}
			}
		}
	}


    int bi, bj, bk;
	for (bk = 0; bk < n_blocks; ++bk) {
		//const int k = bk * BLOCK_SIZE;
		for (bj = 0; bj < n_blocks; ++bj) {
			//const int j = bj * BLOCK_SIZE;
			for (bi = 0; bi < n_blocks; ++bi) {
				//const int i = bi * BLOCK_SIZE;
                do_block(M, Anew, Bnew, C, bi, bj, bk);
            }
        }
    }
}

