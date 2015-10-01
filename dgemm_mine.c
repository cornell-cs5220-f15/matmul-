const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 104) // Multiples of 8 bytes!
#endif

#ifndef SUBBLOCK_SIZE
#define SUBBLOCK_SIZE ((int) 4) // Multiples of BLOCK_SIZE!
#endif

#define NUM_SUBBLOCKS BLOCK_SIZE / SUBBLOCK_SIZE

#include <string.h> //memset

void sublock_dgemm(const double *restrict A, 
                   const double *restrict B, 
                   double *restrict C)
{
    int i, j, k;
    for (j = 0; j < SUBBLOCK_SIZE; ++j) {
        for (k = 0; k < SUBBLOCK_SIZE; ++k) {
            double bkj = B[j*BLOCK_SIZE + k];
            // Unrolled for SUBBLOCK_SIZE 4 to assist icc auto-vectorization
            C[j*BLOCK_SIZE] += A[k*BLOCK_SIZE] * bkj;
            C[j*BLOCK_SIZE + 1] += A[k*BLOCK_SIZE + 1] * bkj;
            C[j*BLOCK_SIZE + 2] += A[k*BLOCK_SIZE + 2] * bkj;
            C[j*BLOCK_SIZE + 3] += A[k*BLOCK_SIZE + 3] * bkj;
        }
    }
}

void do_block(const double *restrict A, 
              const double *restrict B, 
              double *restrict C)
{   
    int bi, bj, bk;
    for (bj = 0; bj < NUM_SUBBLOCKS; ++bj) {
        const int j = bj * SUBBLOCK_SIZE;
        for (bk = 0; bk < NUM_SUBBLOCKS; ++bk) {
            const int k = bk * SUBBLOCK_SIZE;
            const double* Bkj = B + k + j*BLOCK_SIZE;
            for (bi = 0; bi < NUM_SUBBLOCKS; ++bi) {
                const int i = bi * SUBBLOCK_SIZE;
                sublock_dgemm( A + i + k*BLOCK_SIZE,
                               Bkj,
                               C + i + j*BLOCK_SIZE );
            }
        }
    }
}

// Reallocates a (num_rows x num_columns) submatrix with LDA lda
// from src, to a matrix with LDA BLOCK_SIZE
void realloc_block(const int lda,
                   const double *restrict src, double *restrict dest, 
                   const int num_rows, const int num_columns) 
{
    int i, j;
    for (i = 0; i < num_rows; ++i) {
        for (j = 0; j < num_columns; ++j) {
            dest[j*BLOCK_SIZE + i] = src[j*lda + i];
        }
    }
}

void square_dgemm(const int lda,
                  const double *restrict A, 
                  const double *restrict B, 
                  double *restrict C)
{

    // The working arrays for copy optimization
    // All 3 should fit into L2 cache!!
    static double A_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));
    static double B_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));
    static double C_temp[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

    const int n_blocks = lda / BLOCK_SIZE + (lda % BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);

        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);

            memset(C_temp, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    
                memset(A_temp, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
                memset(B_temp, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
                realloc_block(lda, A + i + k*lda, A_temp, M, K);
                realloc_block(lda, B + k + j*lda, B_temp, K, N);
                    
                do_block(A_temp, B_temp, C_temp);
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
