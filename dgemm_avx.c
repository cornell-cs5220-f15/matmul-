#include "immintrin.h"
#include <string.h>

const char* dgemm_desc = "AVX + copy + padded  dgemm.";

#ifndef REG_SIZE
#define REG_SIZE ((int) 4)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void dgemm_4x4(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
	//A, B, C are pointers to the relative positions

	__m256d c_0 = _mm256_load_pd(C + 0);
	__m256d c_1 = _mm256_load_pd(C + 1*lda);
	__m256d c_2 = _mm256_load_pd(C + 2*lda);
	__m256d c_3 = _mm256_load_pd(C + 3*lda);

	//4 columns of A matrix
	__m256d a_k;
	//4 corresponding rows of matrix B
	__m256d b_0k, b_1k, b_2k, b_3k;
	//temp variables to store dot products
	__m256d ctemp_0, ctemp_1, ctemp_2, ctemp_3;

	for (int k = 0; k < K; ++k)	{
		a_k = _mm256_load_pd(A + k*4);

		b_0k = _mm256_broadcast_sd(B + k);
		b_1k = _mm256_broadcast_sd(B + k + lda);
		b_2k = _mm256_broadcast_sd(B + k + 2*lda);
		b_3k = _mm256_broadcast_sd(B + k + 3*lda);

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
	_mm256_store_pd(C+1*lda, c_1);
	_mm256_store_pd(C+2*lda, c_2);
	_mm256_store_pd(C+3*lda, c_3);

}


void block_copy(int M, const double* X, int M_REG, double* XA)	{
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


void reverse_copy(int M, double* X, int M_REG, double* XC)	{
	for (int i = 0; i < M; ++i)	{
		for (int j = 0; j < M; ++j)	{
			X[j + i*M] = XC[j + i*M_REG];
		}
	}
}


void tile_copy(int K, int K_REG, int bi, const double* X, int tile_size, double* XC)	{
	for (int i = 0; i < K_REG; ++i)	{
		for (int j = 0; j < tile_size; ++j)	{
			if ((i < K) && (j+bi < K))	{
				XC[j + i*tile_size] = X[j+bi + i*K];
			} else	{
				XC[j + i*tile_size] = 0;
			}
		}
	}
}


void square_dgemm(const int M, const double* A, const double* B, double* C)
{

	int M_REG = ( M/REG_SIZE + (M%REG_SIZE ? 1 : 0) ) * REG_SIZE;

	int bi, bj;

	//copying into aligned blocks
	double* Bk = _mm_malloc(M_REG * M_REG * sizeof(double), 32);
	double* Ck = _mm_malloc(M_REG * M_REG * sizeof(double), 32);
	double* Ak = _mm_malloc(4 * M_REG * sizeof(double), 32);

	block_copy(M, B, M_REG, Bk);
	//block_copy(M, C, M_REG, Ck);
	memset(Ck, 0, M_REG * M_REG * sizeof(double));


	for (bi = 0; bi < M_REG; bi += 4)	{
		//copying (4-by-M) tile into contiguous, aligned memory
		tile_copy(M, M_REG, bi, A, 4, Ak);

		for (bj = 0; bj < M_REG; bj += 4)	{
			dgemm_4x4(M_REG, M_REG, M_REG, M_REG, Ak, Bk + bj*M_REG, Ck + bi + bj*M_REG);

		}
	
	}

	//copy back into original matrix
	reverse_copy(M, C, M_REG, Ck);

	_mm_free(Ak);
	_mm_free(Bk);
	_mm_free(Ck);

}

