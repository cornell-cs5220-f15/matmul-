const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 8)
#endif

#ifndef BIG_BLOCK_SIZE
#define BIG_BLOCK_SIZE ((int) 96)
#endif
//
/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
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
    const int M = (i+SMALL_BLOCK_SIZE > lda? lda-i : SMALL_BLOCK_SIZE);
    const int N = (j+SMALL_BLOCK_SIZE > lda? lda-j : SMALL_BLOCK_SIZE);
    const int K = (k+SMALL_BLOCK_SIZE > lda? lda-k : SMALL_BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void small_square_dgemm(const int lda, const double *restrict A, const double *restrict B, double *restrict C, const int coarse_i, const int coarse_j, const int coarse_k)
// Are these integer arguments i,j,k constant?
{
    const int n_small_blocks = BIG_BLOCK_SIZE/SMALL_BLOCK_SIZE; // Want to ensure that BIG_BLOCK_SIZE is a multiple of SMALL_BLOCK_SIZE
    // maybe don't need to send n_big_blocks to this function

    const int num_block_M = (coarse_i + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_i) / SMALL_BLOCK_SIZE + ((lda - coarse_i)%SMALL_BLOCK_SIZE? 1 : 0));
    const int num_block_N = (coarse_j + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_j) / SMALL_BLOCK_SIZE + ((lda - coarse_j)%SMALL_BLOCK_SIZE? 1 : 0));
    const int num_block_K = (coarse_k + BIG_BLOCK_SIZE <= lda? n_small_blocks : (lda - coarse_k) / SMALL_BLOCK_SIZE + ((lda - coarse_k)%SMALL_BLOCK_SIZE? 1 : 0));

    //const int num_block_M = (coarse_i < (n_big_blocks-1) * BIG_BLOCK_SIZE? n_small_blocks : (lda - coarse_i) / SMALL_BLOCK_SIZE + ((lda - coarse_i)%SMALL_BLOCK_SIZE? 1 : 0));
    //const int num_block_N = (coarse_j < (n_big_blocks-1) * BIG_BLOCK_SIZE? n_small_blocks : (lda - coarse_j) / SMALL_BLOCK_SIZE + ((lda - coarse_j)%SMALL_BLOCK_SIZE? 1 : 0));
    //const int num_block_K = (coarse_k < (n_big_blocks-1) * BIG_BLOCK_SIZE? n_small_blocks : (lda - coarse_k) / SMALL_BLOCK_SIZE + ((lda - coarse_k)%SMALL_BLOCK_SIZE? 1 : 0));

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
    const int n_big_blocks = M / BIG_BLOCK_SIZE + (M%BIG_BLOCK_SIZE? 1 : 0);
    int big_bi, big_bj, big_bk;
    for (big_bj = 0; big_bj < n_big_blocks; ++big_bj) {
		const int coarse_j = big_bj * BIG_BLOCK_SIZE;
        for (big_bk = 0; big_bk < n_big_blocks; ++big_bk) {
            const int coarse_k = big_bk * BIG_BLOCK_SIZE;
            for (big_bi = 0; big_bi < n_big_blocks; ++big_bi) {
				const int coarse_i = big_bi * BIG_BLOCK_SIZE;
                //do_block(M, A, B, C, i, j, k);
                small_square_dgemm(M, A, B, C, coarse_i, coarse_j, coarse_k);
            }
        }
    }
}