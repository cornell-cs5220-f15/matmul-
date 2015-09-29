#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char* dgemm_desc = "copy-blocked dgemm.";

#ifndef MUL_BLOCK_SIZE
  #define MUL_BLOCK_SIZE ((int) 4)
#endif

#ifndef TRANSPOSE_BLOCK_SIZE
  #define TRANSPOSE_BLOCK_SIZE ((int) 16)
#endif

#ifndef MEM_BOUNDARY
  #define MEM_BOUNDARY ((int) 32)
#endif

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

inline void transpose_scalar_block(const double *A, double *B, const int lda, const int M)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);

  int row, col;

  #pragma omp parallel for
  for (col = 0; col < M; col++)
  {
    for (row = 0; row < M; row++)
    {
      B[row*lda + col] = A[col*lda + row];
    }
  }
}

inline void transpose_block(const double *A, double *B, const int M)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);

  int row, col;
  int lda = ROUND_UP(M, TRANSPOSE_BLOCK_SIZE);

  #pragma omp parallel for
  for (col = 0; col < M; col += TRANSPOSE_BLOCK_SIZE)
  {
    for (row = 0; row < M; row += TRANSPOSE_BLOCK_SIZE)
    {
      transpose_scalar_block(&A[row*lda + col], &B[col*lda + row], lda, TRANSPOSE_BLOCK_SIZE);
    }
  }
}

inline void transpose_array(const int M, const double *A, double *B)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);

  int row, col;

  for (col = 0; col < M; col++)
  {
    for (row = 0; row < M; row++)
    {
      B[(row * M) + col] = A[(col * M) + row];
    }
  }
}

/**
 * Multiplies a pair of 4 x 4 matrices using AVX instructions.
 * Requires that all matrix inputs be aligned to 32 bytes or shit hits the fan
 * @method dgemm_4x4
 * @param  A         4 x 4 matrix in row-order
 * @param  B         4 x 4 matrix in column-order
 * @param  C         4 x 4 matrix in column-order
 */
inline void dgemm_4x4(const double * restrict A, const double * restrict B, double * restrict C)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);
  __assume_aligned(C, MEM_BOUNDARY);

  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;

  printf("A:\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n\n", *A, *(A + 1), *(A + 2), *(A + 3), *(A + 4), *(A + 5), *(A + 6), *(A + 7), *(A + 8), *(A + 9), *(A + 10), *(A + 11), *(A + 12), *(A + 13), *(A + 14), *(A + 15));

  printf("B:\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n\n", *B, *(B + 4), *(B + 8), *(B + 12), *(B + 1), *(B + 5), *(B + 9), *(B + 13), *(B + 2), *(B + 6), *(B + 10), *(B + 14), *(B + 3), *(B + 7), *(B + 11), *(B + 15));

  double *a_transposed = _mm_malloc(16 * sizeof(double), MEM_BOUNDARY);
  transpose_array(4, A, a_transposed);

  // Load columns of A
  ymm0 = _mm256_load_pd(a_transposed);
  ymm1 = _mm256_load_pd(a_transposed + 4);
  ymm2 = _mm256_load_pd(a_transposed + 8);
  ymm3 = _mm256_load_pd(a_transposed + 12);

  // Broadcast B11, B21, B31, and B41
  ymm4 = _mm256_broadcast_sd(B);
  ymm5 = _mm256_broadcast_sd(B + 1);
  ymm6 = _mm256_broadcast_sd(B + 2);
  ymm7 = _mm256_broadcast_sd(B + 3);

  // Calculate dot products
  ymm4 = _mm256_mul_pd(ymm0, ymm4);
  ymm5 = _mm256_mul_pd(ymm1, ymm5);
  ymm6 = _mm256_mul_pd(ymm2, ymm6);
  ymm7 = _mm256_mul_pd(ymm3, ymm7);

  // Sum to get first column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  _mm256_store_pd(C, ymm4);

  // Broadcast B12, B22, B32, and B42
  ymm4 = _mm256_broadcast_sd(B + 4);
  ymm5 = _mm256_broadcast_sd(B + 5);
  ymm6 = _mm256_broadcast_sd(B + 6);
  ymm7 = _mm256_broadcast_sd(B + 7);

  // Calculate dot products
  ymm4 = _mm256_mul_pd(ymm0, ymm4);
  ymm5 = _mm256_mul_pd(ymm1, ymm5);
  ymm6 = _mm256_mul_pd(ymm2, ymm6);
  ymm7 = _mm256_mul_pd(ymm3, ymm7);

  // Sum to get first column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm4 = _mm256_add_pd(ymm4, ymm6);
  ymm4 = _mm256_add_pd(ymm4, ymm7);

  // Writeback to memory
  _mm256_store_pd(C + 4, ymm4);

  // Broadcast B13, B23, B33, and B43
  ymm4 = _mm256_broadcast_sd(B + 8);
  ymm5 = _mm256_broadcast_sd(B + 9);
  ymm6 = _mm256_broadcast_sd(B + 10);
  ymm7 = _mm256_broadcast_sd(B + 11);

  // Calculate dot products
  ymm4 = _mm256_mul_pd(ymm0, ymm4);
  ymm5 = _mm256_mul_pd(ymm1, ymm5);
  ymm6 = _mm256_mul_pd(ymm2, ymm6);
  ymm7 = _mm256_mul_pd(ymm3, ymm7);

  // Sum to get first column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  _mm256_store_pd(C + 8, ymm4);

  // Broadcast B14, B24, B34, and B44
  ymm4 = _mm256_broadcast_sd(B + 12);
  ymm5 = _mm256_broadcast_sd(B + 13);
  ymm6 = _mm256_broadcast_sd(B + 14);
  ymm7 = _mm256_broadcast_sd(B + 15);

  // Calculate dot products
  ymm4 = _mm256_mul_pd(ymm0, ymm4);
  ymm5 = _mm256_mul_pd(ymm1, ymm5);
  ymm6 = _mm256_mul_pd(ymm2, ymm6);
  ymm7 = _mm256_mul_pd(ymm3, ymm7);

  // Sum to get first column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  _mm256_store_pd(C + 12, ymm4);

  printf("C:\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n\n", *C, *(C + 4), *(C + 8), *(C + 12), *(C + 1), *(C + 5), *(C + 9), *(C + 13), *(C + 2), *(C + 6), *(C + 10), *(C + 14), *(C + 3), *(C + 7), *(C + 11), *(C + 15));
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / MUL_BLOCK_SIZE + (M % MUL_BLOCK_SIZE? 1 : 0);
    int blocked_row, blocked_col, blocked_k, row_start, col_start, k_start, row, col, k, col_size, row_size, k_size;

    // Create a copy of A that is transposed in row-major order
    // to get element at (col,row): a_transposed[row*M + col]
    double *a_transposed = _mm_malloc(M * M * sizeof(double), MEM_BOUNDARY);
    transpose_array(M, A, a_transposed);

    int malloc_block_size = MUL_BLOCK_SIZE * MUL_BLOCK_SIZE * sizeof(double);

    double* a_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* b_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* c_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);

    for (blocked_col = 0; blocked_col < n_blocks; blocked_col++)
    {
      col_start = blocked_col * MUL_BLOCK_SIZE;
      col_size =  ((col_start + MUL_BLOCK_SIZE) > M) ? (M - col_start) : MUL_BLOCK_SIZE;

      for (blocked_row = 0; blocked_row < n_blocks; blocked_row++)
      {
        row_start = blocked_row * MUL_BLOCK_SIZE;
        row_size = ((row_start + MUL_BLOCK_SIZE) > M) ? (M - row_start) : MUL_BLOCK_SIZE;

  			// Reset C to zero
        c_block = memset(c_block, 0, malloc_block_size);

        #pragma vector aligned
        for (blocked_k = 0; blocked_k < n_blocks; blocked_k++)
        {
        	// Reset A and B to zero
          a_block = memset(a_block, 0, malloc_block_size);
          b_block = memset(b_block, 0, malloc_block_size);

          // C = A[blocked_k, blocked_row] * B[blocked_col, blocked_k]
          k_start = blocked_k * MUL_BLOCK_SIZE;
          k_size = ((k_start + MUL_BLOCK_SIZE) > M) ? (M - k_start) : MUL_BLOCK_SIZE;

          // Copying A into a contiguous block of memory
          #pragma omp parallel for
          for (row = 0; row < row_size; row++)
          {
          	int row_start_inner = ((row_start + row) * M) + k_start;

            for (k = 0; k < k_size; ++k)
            {
              a_block[(row * MUL_BLOCK_SIZE) + k] = a_transposed[row_start_inner + k];
            }
          }

          // Copying B into a contiguous block of memory
          #pragma omp parallel for
          for (col = 0; col < col_size; col++)
          {
          	int col_start_inner = ((col_start + col)*M) + k_start;

            for (k = 0; k < k_size; ++k)
            {
              b_block[(col * MUL_BLOCK_SIZE) + k] =  B[col_start_inner + k];
            }
          }

          // Write into temporary array C assuming A is row-major and B is column-major
          dgemm_4x4(a_block, b_block, c_block);
        }

        // Write c_block into actual location in C
        #pragma omp parallel for
        for (col = 0; col < col_size; col++)
        {
          for (row = 0; row < row_size; row++)
          {
            C[((col_start + col) * M) + row_start + row] = c_block[col*MUL_BLOCK_SIZE + row];
          }
        }
      }
    }

    // Clean up in order of initialization
    _mm_free(c_block);
    _mm_free(b_block);
    _mm_free(a_block);
    _mm_free(a_transposed);
}
