#include "immintrin.h"
#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "AVX + copy + padded + blocked dgemm.";

#ifndef REGISTER_SIZE
#define REGISTER_SIZE ((int) 4)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 768)
#endif

/*
  lda is the leading dimension of matrix A, ldb is the leading dimension of matrix B,
  and ldc is the leading dimension of matrix C.
*/
void dgemm_4x4(const int mult_a, const int lda, const int ldb, const int ldc,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
	//A, B, C are pointers to the relative positions where the blocks begin

	//4 columns of matrix C
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

	//k is the row of B and the column of A
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


void copy_and_pad(const int M, const double* restrict X, const int M_padded, double* restrict X_aligned)	{
	//copies matrix to a new location with zero padding
	for (int i = 0; i < M_padded; ++i)	{
		for (int j = 0; j < M_padded; ++j)	{
			if ((i < M) && (j < M))	{
				X_aligned[j + i*M_padded] = X[j + i*M];
			} else	{
				X_aligned[j + i*M_padded] = 0;
			}
		}	
	}

}


void copy_unpad(const int M, double* restrict X, const int M_padded, double* restrict X_padded)	{
	for (int i = 0; i < M; ++i)	{
		for (int j = 0; j < M; ++j)	{
			X[j + i*M] = X_padded[j + i*M_padded];
		}
	}
}


void tile_copy( const int lda, const double* restrict Xrel, const int j_offset,
		const int i_offset, const int tile_width, const int tile_height, double* restrict X_tile)
{
	//Xrel is a relative pointer into X, at offset given by j_offset and i_offset
	for (int i = 0; i < tile_height; ++i)	{
		for (int j = 0; j < tile_width; ++j)	{
			if ((j_offset + j < lda) && (i_offset + i < lda))	{
				X_tile[j + i*tile_width] = Xrel[j + i*lda];
			} else	{
				X_tile[j + i*tile_width] = 0;
			}
		}
	}
}


//In the inputs to this function, B and C are already aligned/padded, but A is not.
void do_block(const int lda, const int lda_old, const double* restrict A, 
		const double* restrict B, double* restrict C, const int j, const int k )
{
	//const int M = (i+BLOCK_SIZE > lda ? lda-i : BLOCK_SIZE);
	const int K = (k+BLOCK_SIZE > lda ? lda-k : BLOCK_SIZE);
	const int N = (j+BLOCK_SIZE > lda ? lda-j : BLOCK_SIZE);

	//const int Arel = i + k*lda_old;
	//const int Brel = k + j*lda;
	//const int Crel = i + j*lda;

	//Offsets to the start of the current block's column and/or row
	const int A_col_offset = k*lda_old;
	const int B_block_offset = k + j*lda;
	const int C_col_offset = j*lda;

	//Aligned memory for an M-by-K slice of A
	double* A_aligned = _mm_malloc(lda * K * sizeof(double), 32);

	//unrolling loops by a factor of 4 for rows and columns of C (M-by-N)
	int bi, bj;
	for (bj = 0; bj < N; bj += 4)	{

		for (bi = 0; bi < lda; bi += 4)	{

			//copying (4-by-K) tile into contiguous, aligned memory
			if (bj == 0) {
				tile_copy(lda_old, A + A_col_offset + bi, bi, k, 
						REGISTER_SIZE, K, A_aligned + bi*K);
			}
			dgemm_4x4(K, REGISTER_SIZE, lda, lda, 
					A_aligned + bi*K, B + B_block_offset + bj*lda, C + C_col_offset + bi + bj*lda);
		}	
	}
	_mm_free(A_aligned);
}


void square_dgemm(const int M, const double* A, const double* B, double* C)
{

	//Round up M to the nearest multiple of the register size
	const int M_padded = ( M/REGISTER_SIZE + (M%REGISTER_SIZE ? 1 : 0) ) * REGISTER_SIZE;

	//Allocate memory for aligned copies
	double* B_aligned = _mm_malloc(M_padded * M_padded * sizeof(double), 32);
	double* C_aligned = _mm_malloc(M_padded * M_padded * sizeof(double), 32);
	
	copy_and_pad(M, B, M_padded, B_aligned);
	memset(C_aligned, 0, M_padded * M_padded * sizeof(double));

	const int n_blocks = M_padded / BLOCK_SIZE + (M_padded%BLOCK_SIZE ? 1 : 0);
	int bi, bj, bk;
	for (bj = 0; bj < n_blocks; ++bj)	{
		const int j = bj * BLOCK_SIZE;
		for (bk = 0; bk < n_blocks; ++bk)	{
			const int k = bk * BLOCK_SIZE;
				//for (bi = 0; bi < n_blocks; ++bi)	{
				//	const int i = bi * BLOCK_SIZE;
					do_block(M_padded, M, A, B_aligned, C_aligned, j, k);
				//}
		}
	}

	//copy back into original matrix
	copy_unpad(M, C, M_padded, C_aligned);

	_mm_free(B_aligned);
	_mm_free(C_aligned);
}
