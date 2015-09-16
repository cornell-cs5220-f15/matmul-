const char* dgemm_desc = "Blocked dgemm with copy optimization (transposing the copy). Not yet finished. Inside loop ordering is j,k,i.";

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
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            double bkj = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                C[j*lda+i] += A[k*lda+i] * B[j*lda+k];
            }
        }
    }
}

/* Copies a block of matrix D starting at (i,j) into a new block of memory 
(TODO: possibly padding it with zeros) and returns a pointer to it*/
const double* copy_block(const int lda, const int i, const int j, const double *D)
{
    double *D_block = (double *)malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    int k,l;
    for(l = 0; l < BLOCK_SIZE; ++l) {
        for(k = 0; k < BLOCK_SIZE; ++k) {
        	D_block[l*BLOCK_SIZE+k] = D[(j+l)*lda+(i+k)];
        }
    }
    return D_block;
}

/* replaces the block in C with C_block */
void replace_block(const int lda, const double *C, const double *C_block,const int i,const int j) {
    int k,l;
    for(l = 0; l < BLOCK_SIZE; ++l) {
        for(k = 0; k < BLOCK_SIZE; ++k) {
        	C[(j+l)*lda+(i+k)] = C_block[l*BLOCK_SIZE+k];
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
	if(i+BLOCK_SIZE > lda || j+BLOCK_SIZE > lda || k+BLOCK_SIZE > lda) {
		const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
		const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
		const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
		basic_dgemm(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
    } else {
    	const double *A_block = copy_block(lda,i,k,*A);
    	const double *B_block = copy_block(lda,k,j,*B);
    	const double *C_block = copy_block(lda,i,j,*C);
    	basic_dgemm(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE,A_block,B_block,C_block);
    	replace_block(lda,C,C_block);
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bk = 0; bk < n_blocks; ++bk) {
        const int k = bk * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}



