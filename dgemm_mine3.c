const char* dgemm_desc = "My dgemm with copy optimization";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif
#ifndef ALIGNED_SIZE
#define ALIGNED_SIZE ((int) 64)
#endif


double* copy_optimize(const int M, const int n_blocks, const double* A )
{
    int Mc = BLOCK_SIZE * n_blocks;
    double* cp = (double*) _mm_malloc(Mc * Mc * sizeof( double) , ALIGNED_SIZE);
    int i, j, I, J, ii, jj, id;
    memset(cp, 0, Mc * Mc * sizeof( double));
    for (i = 0; i < M; ++i)
    {
    	I = i / BLOCK_SIZE;
    	ii = i % BLOCK_SIZE;
        for (j = 0; j < M; ++j)
        {
        	J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            id = (I * n_blocks + J) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            cp[id] = A[ j * M + i ];
        }
    }
    return cp;
}

void copy_back(const int M, const int n_blocks, const double* A, double* cp)
{
    int i, j, I, J, ii, jj, id;
    for (i = 0; i < M; ++i)
    {
    	I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for (j = 0; j < M; ++j)
        {
        	J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            id = (I * n_blocks + J) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            cp[j * M + i] = A[id];
        }
    }
}

void do_block(const double* A_block, const double* B_block, double* C_block)
{
	int i, j, k;
	__assume_aligned( A_block, ALIGNED_SIZE );
    __assume_aligned( B_block, ALIGNED_SIZE );
    __assume_aligned( C_block, ALIGNED_SIZE );
    for (i = 0; i < BLOCK_SIZE; ++i)
    {
    	double* Ci = C_block + i * BLOCK_SIZE;
    	__assume_aligned(Ci, ALIGNED_SIZE);
    	for (k = 0; k < BLOCK_SIZE; ++k)
    	{
    		double Aik = A_block[i * BLOCK_SIZE + k];
    		double* Bk = B_block + k * BLOCK_SIZE;
    		__assume_aligned(Bk, ALIGNED_SIZE);
    		for (j = 0; j < BLOCK_SIZE; ++j)
    		{
    			//#pragma vector always
    			Ci[j] += Aik * Bk[j];
    		}
    	}
    }
}


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    double* A_cp = copy_optimize(M, n_blocks, A);
    double* B_cp = copy_optimize(M, n_blocks, B);
    double* C_cp = copy_optimize_rm(M, n_blocks, C);
    double *A_block, *B_block, *C_block;

    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                A_block = A_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * n_blocks + bk);
                B_block = B_cp + BLOCK_SIZE * BLOCK_SIZE * (bk * n_blocks + bj);
                C_block = C_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * n_blocks + bj);
                do_block(A_block, B_block, C_block);
            }
        }
    }
    copy_back(M, n_blocks, C_cp, C);
    _mm_free(A_cp);
    _mm_free(B_cp);
    _mm_free(C_cp);
}
