const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64) // Multiples of 8 bytes!
#endif

void block_dgemm(const double *restrict A, 
                 const double *restrict B, 
                 double *restrict C)
{
    int i, j, k;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        for (k = 0; k < BLOCK_SIZE; ++k) {
            double bkj = B[j*BLOCK_SIZE + k];
            for (i = 0; i < BLOCK_SIZE; ++i) {
                C[j*BLOCK_SIZE + i] += A[k*BLOCK_SIZE + i] * bkj;
            }
        }
    }
}

void realloc_block(const int lda1, const int lda2,
                   const double *restrict A1, double *restrict A2, 
                   const int r_limit, const int c_limit) 
{
    int i, j;
    for (i = 0; i < lda2; ++i) {
        for (j = 0; j < lda2; ++j) {
            A2[j*lda2 + i] = (i < r_limit && j < c_limit) ? A1[j*lda1 + i] : 0;
        }
    }
}

void square_dgemm(const int lda,
                  const double *restrict A, 
                  const double *restrict B, 
                  double *restrict C)
{

    // The working arrays for copy optimization
    double A_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));
    double B_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));
    double C_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

    const int n_blocks = lda / BLOCK_SIZE + (lda % BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);

            realloc_block(lda, BLOCK_SIZE, C + i + j*lda, C_temp, M, N);

            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
                
                realloc_block(lda, BLOCK_SIZE, A + i + k*lda, A_temp, M, K);
                realloc_block(lda, BLOCK_SIZE, B + k + j*lda, B_temp, K, N);
                
                block_dgemm(A_temp, B_temp, C_temp);
            }

            // copy results back into main C array
            int ci, cj;
            for (ci = 0; ci < M; ++ci) {
                for (cj = 0; cj < N; ++cj) {
                    C[(j + cj)*lda + (i+ci)] = C_temp[cj*BLOCK_SIZE + ci];
                }
            }
        }
    }
}
