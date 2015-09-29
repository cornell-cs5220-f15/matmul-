#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* dgemm_desc = "avx copy-blocked dgemm.";

#ifndef MUL_BLOCK_SIZE
  #define MUL_BLOCK_SIZE ((int) 4)
#endif

#ifndef MEM_BOUNDARY
  #define MEM_BOUNDARY ((int) 32)
#endif

inline void transpose_array(const int M, const double * restrict A, double * restrict B)
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
 *
 * Broadcasts an element in a column of B into a ymm register, and multiplies
 * by a column of A. Doing this for all four columns of A gives the components
 * of the dot product, which are then summed to produce a column in C.
 *
 * e.g. for the first column of C:
 * A_11 B_11 + A_12 B_21 + A_13 B_31 + A_14 B_41 = C_11
 * A_21 B_11 + A_22 B_21 + A_23 B_31 + A_24 B_41 = C_21
 * A_31 B_11 + A_32 B_21 + A_33 B_31 + A_34 B_41 = C_31
 * A_41 B_11 + A_42 B_21 + A_43 B_31 + A_44 B_41 = C_41
 * ---- ----   ---- ----   ---- ----   ---- ----
 * ymm1 ymm4   ymm2 ymm5   ymm3 ymm6   ymm4 ymm7
 *
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

  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm9;

  // Convert A to column order
  // TODO: Switch the input to be in column order so we don't waste time mallocing
  double *a_transposed = _mm_malloc(4 * 4 * sizeof(double), MEM_BOUNDARY);
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
  ymm9 = _mm256_load_pd(C);
  ymm4 = _mm256_add_pd(ymm4, ymm9);
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

  // Sum to get second column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  ymm9 = _mm256_load_pd(C + 4);
  ymm4 = _mm256_add_pd(ymm4, ymm9);
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

  // Sum to get third column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  ymm9 = _mm256_load_pd(C + 8);
  ymm4 = _mm256_add_pd(ymm4, ymm9);
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

  // Sum to get fourth column of C
  ymm4 = _mm256_add_pd(ymm4, ymm5);
  ymm6 = _mm256_add_pd(ymm6, ymm7);
  ymm4 = _mm256_add_pd(ymm4, ymm6);

  // Writeback to memory
  ymm9 = _mm256_load_pd(C + 12);
  ymm4 = _mm256_add_pd(ymm4, ymm9);
  _mm256_store_pd(C + 12, ymm4);

  _mm_free(a_transposed);
}

inline void dgemm_8x8(const double * restrict A, const double * restrict B, double * restrict C)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);
  __assume_aligned(C, MEM_BOUNDARY);

  dgemm_4x4(A,      B,      C);
  dgemm_4x4(A + 32, B,      C + 16);
  dgemm_4x4(A,      B + 32, C + 32);
  dgemm_4x4(A + 32, B + 32, C + 48);

  // TODO: Convert the matrix back to correct layout
}

inline void dgemm_16x16(const double * restrict A, const double * restrict B, double * restrict C)
{

}

inline void dgemm_32x32(const double * restrict A, const double * restrict B, double * restrict C)
{

}

inline void dgemm_64x64(const double * restrict A, const double * restrict B, double * restrict C)
{

}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / MUL_BLOCK_SIZE + (M % MUL_BLOCK_SIZE? 1 : 0);
    int blocked_row, blocked_col, blocked_k, row_start, col_start, k_start, row, col, k, col_size, row_size, k_size;

    // Create a copy of A that is transposed in row-major order
    // to get element at (col,row): a_transposed[row * M + col]
    double *a_transposed = _mm_malloc(M * M * sizeof(double), MEM_BOUNDARY);
    transpose_array(M, A, a_transposed);

    int malloc_block_size = MUL_BLOCK_SIZE * MUL_BLOCK_SIZE * sizeof(double);

    double* a_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* b_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* c_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);

    for (blocked_col = 0; blocked_col < n_blocks; blocked_col++)
    {
      col_start = blocked_col * MUL_BLOCK_SIZE;
      col_size = ((col_start + MUL_BLOCK_SIZE) > M) ? (M - col_start) : MUL_BLOCK_SIZE;

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
          for (row = 0; row < row_size; row++)
          {
          	int row_start_inner = ((row_start + row) * M) + k_start;

            for (k = 0; k < k_size; ++k)
            {
              a_block[(row * MUL_BLOCK_SIZE) + k] = a_transposed[row_start_inner + k];
            }
          }

          // Copying B into a contiguous block of memory
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
        for (col = 0; col < col_size; col++)
        {
          for (row = 0; row < row_size; row++)
          {
            C[((col_start + col) * M) + row_start + row] = c_block[col * MUL_BLOCK_SIZE + row];
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
