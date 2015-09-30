#include "immintrin.h"
#include <string.h>

const char* dgemm_desc = "AVX + copy + padded  dgemm.";

#ifndef REGISTER_SIZE
#define REGISTER_SIZE ((int) 4)
#endif

/*
  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void dgemm_4x4(const int lda, const double * restrict A,
		const double * restrict B, double * restrict C)
{
	//A, B, C are pointers to the relative positions where the blocks begin

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

	for (int k = 0; k < lda; ++k)	{
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


void copy_and_pad(const int M, const double* X, const int M_padded, double* X_aligned)	{
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


void copy_unpad(const int M, double* X, const int M_padded, const double* X_aligned)	{
	for (int i = 0; i < M; ++i)	{
		for (int j = 0; j < M; ++j)	{
			X[j + i*M] = X_aligned[j + i*M_padded];
		}
	}
}


void tile_copy_and_pad(const int M, const double* restrict X, const int M_padded, const int X_col_offset, const int tile_width, double* restrict X_tile)	{
	for (int i = 0; i < M_padded; ++i)	{
		for (int j = 0; j < tile_width; ++j)	{
			if ((i < M) && (j+X_col_offset < M))	{
				X_tile[j + i*tile_width] = X[j+X_col_offset + i*M];
			} else	{
				X_tile[j + i*tile_width] = 0;
			}
		}
	}
}


void square_dgemm(const int M, const double* A, const double* B, double* C)
{

	//Round up M to the nearest multiple of the register size
	const int M_padded = ( M/REGISTER_SIZE + (M%REGISTER_SIZE ? 1 : 0) ) * REGISTER_SIZE;

	int block_i_offset, block_j_offset;

	//Allocate memory for aligned copies
	double* restrict B_aligned = _mm_malloc(M_padded * M_padded * sizeof(double), 32);
	double* restrict C_aligned = _mm_malloc(M_padded * M_padded * sizeof(double), 32);
	double* restrict A_tile = _mm_malloc(4 * M_padded * sizeof(double), 32);

	//copy B into aligned memory with padding
	copy_and_pad(M, B, M_padded, B_aligned);
	memset(C_aligned, 0, M_padded * M_padded * sizeof(double));


	for (block_i_offset = 0; block_i_offset < M_padded; block_i_offset += REGISTER_SIZE) {
		//copy (4-by-M) tile of A into contiguous, aligned memory
		tile_copy_and_pad(M, A, M_padded, block_i_offset, REGISTER_SIZE, A_tile);

		//For each long tile, compute the product with B in 4-by-4 blocks
		for (block_j_offset = 0; block_j_offset < M_padded; block_j_offset += REGISTER_SIZE) {
			dgemm_4x4(M_padded, A_tile, B_aligned + block_j_offset*M_padded, 
					C_aligned + block_i_offset + block_j_offset*M_padded);

		}
	
	}

	//copy output back into original matrix
	copy_unpad(M, C, M_padded, C_aligned);

	_mm_free(A_tile);
	_mm_free(B_aligned);
	_mm_free(C_aligned);

}

