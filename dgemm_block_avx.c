#include "immintrin.h"
#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "AVX + copy + padded + blocked dgemm.";

#ifndef REG_SIZE
#define REG_SIZE ((int) 4)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 768)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void dgemm_4x4(const int mult_a, const int lda, const int ldb, const int ldc,
                 const double *A, const double *B, double *C)
{
	//A, B, C are pointers to the relative positions

	__m256d c_0 = _mm256_load_pd(C + 0);
	__m256d c_1 = _mm256_load_pd(C + 1*ldc);
	__m256d c_2 = _mm256_load_pd(C + 2*ldc);
	__m256d c_3 = _mm256_load_pd(C + 3*ldc);

	//4 columns of A matrix
	__m256d a_k;
	//4 corresponding rows of matrix B
	__m256d b_0k, b_1k, b_2k, b_3k;
	//temp variables to store dot products
	__m256d ctemp_0, ctemp_1, ctemp_2, ctemp_3;

	for (int k = 0; k < mult_a; ++k)	{
		a_k = _mm256_load_pd(A + k*lda);

		b_0k = _mm256_broadcast_sd(B + k);
		b_1k = _mm256_broadcast_sd(B + k + ldc);
		b_2k = _mm256_broadcast_sd(B + k + 2*ldc);
		b_3k = _mm256_broadcast_sd(B + k + 3*ldc);

		ctemp_0 = _mm256_mul_pd(a_k, b_0k);
		ctemp_1 = _mm256_mul_pd(a_k, b_1k);
		ctemp_2 = _mm256_mul_pd(a_k, b_2k);
		ctemp_3 = _mm256_mul_pd(a_k, b_3k);

		c_0 = _mm256_add_pd(c_0, ctemp_0);
		c_1 = _mm256_add_pd(c_1, ctemp_1);
		c_2 = _mm256_add_pd(c_2, ctemp_2);
		c_3 = _mm256_add_pd(c_3, ctemp_3);
	}

	_mm256_store_pd(C+0, c_0);
	_mm256_store_pd(C+1*ldc, c_1);
	_mm256_store_pd(C+2*ldc, c_2);
	_mm256_store_pd(C+3*ldc, c_3);

}


void block_padding(const int M, const double* X, const int M_REG, double* XA)	{
	//copies matrix to a new location with zero padding
	for (int i = 0; i < M_REG; ++i)	{
		for (int j = 0; j < M_REG; ++j)	{
			if ((i < M) && (j < M))	{
				XA[j + i*M_REG] = X[j + i*M];
			} else	{
				XA[j + i*M_REG] = 0;
			}
		}	
	}
}


void reverse_copy(const int M, double* X, const int M_REG, double* XC)	{
	for (int i = 0; i < M; ++i)	{
		for (int j = 0; j < M; ++j)	{
			X[j + i*M] = XC[j + i*M_REG];
		}
	}
}


void tile_copy( const int lda, const int K, const double* Xrel, const int ri,
		const int rk, const int tile_size, double* XC)
{
	//Relative position of X as input
	for (int k = 0; k < K; ++k)	{
		for (int i = 0; i < tile_size; ++i)	{
			if ((ri + i < lda) && (rk + k < lda))	{
				XC[ i + k*tile_size] = Xrel[ i + k*lda];
			} else	{
				XC[ i + k*tile_size] = 0;
			}
		}
	}
}


void do_block(  const int lda, const int ld_old, const double* A, 
		const double* B, double* C, const int j, const int k )
{
	//const int M = (i+BLOCK_SIZE > lda ? lda-i : BLOCK_SIZE);
	const int K = (k+BLOCK_SIZE > lda ? lda-k : BLOCK_SIZE);
	const int N = (j+BLOCK_SIZE > lda ? lda-j : BLOCK_SIZE);

	//const int Arel = i + k*ld_old;
	//const int Brel = k + j*lda;
	//const int Crel = i + j*lda;

	const int Arel = k*ld_old;
	const int Brel = k + j*lda;
	const int Crel = j*lda;

	double* Ak = _mm_malloc(lda * K * sizeof(double), 32);

	//unrolling loops by a factor of 4 for rows and columns of C (M-by-N)
	int bi, bj;
	for (bj = 0; bj < N; bj += 4)	{

		for (bi = 0; bi < lda; bi += 4)	{

			//copying (4-by-K) tile into contiguous, aligned memory
			if (bj == 0)	{
			tile_copy(ld_old, K, A + Arel + bi, bi, k,  4, Ak + bi*K);
			}
			dgemm_4x4(K, 4, lda, lda, Ak + bi*K, B + Brel + bj*lda, C + Crel + bi + bj*lda);
		}	
	}
	_mm_free(Ak);
}


void square_dgemm(const int M, const double* A, const double* B, double* C)
{

	const int M_REG = ( M/REG_SIZE + (M%REG_SIZE ? 1 : 0) ) * REG_SIZE;

	//copying into aligned blocks
	double* Bk = _mm_malloc(M_REG * M_REG * sizeof(double), 32);
	double* Ck = _mm_malloc(M_REG * M_REG * sizeof(double), 32);
	
	block_padding(M, B, M_REG, Bk);
	memset(Ck, 0, M_REG * M_REG * sizeof(double));

	const int n_blocks = M_REG / BLOCK_SIZE + (M_REG%BLOCK_SIZE ? 1 : 0);
	int bi, bj, bk;
	for (bj = 0; bj < n_blocks; ++bj)	{
		const int j = bj * BLOCK_SIZE;
		for (bk = 0; bk < n_blocks; ++bk)	{
			const int k = bk * BLOCK_SIZE;
				//for (bi = 0; bi < n_blocks; ++bi)	{
				//	const int i = bi * BLOCK_SIZE;
					do_block(M_REG, M, A, Bk, Ck, j, k);
				//}
		}
	}

	//copy back into original matrix
	reverse_copy(M, C, M_REG, Ck);

	_mm_free(Bk);
	_mm_free(Ck);
}
