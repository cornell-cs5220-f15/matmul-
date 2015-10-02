const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 512)
#endif

#include<immintrin.h>
#include<stdlib.h>

void basic_dgemm(const int lda, const int M, const int N, const int K,
		const double *A, const double *B, double *C) 
{
	int i, j, k;
	const int col_reduced_4 = M - M % 4;
	// This is used for loop unrolling
	const int col_reduced_16 = M - M % 16;
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8;
	// Memory alignment
	double *b = _mm_malloc(K * sizeof(double), 64);
	double *c = _mm_malloc(M * sizeof(double), 64);
	// Try to do the computation in the jki order
	for (j = 0; j < N; ++j) {
		const int index1 = j * lda;
		for (k = 0; k < K; k++) {
			b[k] = B[index1 + k];
		}
		for (i = 0; i < M; i++) {
			c[i] = C[index1 + i];
		}
		for (k = 0; k < K; ++k) {
			const int index2 = k * lda;
			// broadcast b[k] to 4 double digits that are stored in the 
			// 256 aligned memory
			ymm0 = _mm256_broadcast_sd(&b[k]);
			// Do the C[i] += A[index2 + i] * B[k] using loop unrolling
			for (i = 0; i < col_reduced_16; i += 16) {
				// load in 16 digits of A, since A is not aligned,
				// we need to use loadu.
				ymm1 = _mm256_loadu_pd(&A[index2 + i]);
				ymm2 = _mm256_loadu_pd(&A[index2 + i + 4]);
				ymm3 = _mm256_loadu_pd(&A[index2 + i + 8]);
				ymm4 = _mm256_loadu_pd(&A[index2 + i + 12]);

				// load in 16 digits of C
				ymm5 = _mm256_load_pd(&c[i]);
				ymm6 = _mm256_load_pd(&c[i + 4]);
				ymm7 = _mm256_load_pd(&c[i + 8]);
				ymm8 = _mm256_load_pd(&c[i + 12]);

				// This line is doing C[i] += A[index2 + i] * B[k] using fma
				_mm256_store_pd(&c[i], _mm256_fmadd_pd(ymm1, ymm0, ymm5));
				_mm256_store_pd(&c[i + 4], _mm256_fmadd_pd(ymm2, ymm0, ymm6));
				_mm256_store_pd(&c[i + 8], _mm256_fmadd_pd(ymm3, ymm0, ymm7));
				_mm256_store_pd(&c[i + 12], _mm256_fmadd_pd(ymm4, ymm0, ymm8));
			}
			for (i = col_reduced_16; i < M; i++) {
				c[i] += A[index2 + i] * b[k];
			}
		}
		for (i = 0; i < col_reduced_4; i += 4) {
			_mm256_storeu_pd(&C[index1 + i], _mm256_loadu_pd(&c[i]));
		}
		for (i = col_reduced_4; i < M; i++) {
			C[index1 + i] = c[i];
		}
	}   
	_mm_free(b);
	_mm_free(c);
}

void do_block(const int lda,
		const double *A, const double *B, double *C,
		const int i, const int j, const int k)
{
	const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
	const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
	const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
	basic_dgemm(lda, M, N, K,
			A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
	const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
	int bi, bj, bk;
	for (bi = 0; bi < n_blocks; ++bi) {
		const int i = bi * BLOCK_SIZE;
		for (bj = 0; bj < n_blocks; ++bj) {
			const int j = bj * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks; ++bk) {
				const int k = bk * BLOCK_SIZE;
				do_block(M, A, B, C, i, j, k);
			}
		}
	}
}

