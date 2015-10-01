#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const char* dgemm_desc = "Copy optimized, blocked dgemm using an AVX kernel";

#ifndef MUL_BLOCK_SIZE
  #define MUL_BLOCK_SIZE ((int) 8)
#endif

#ifndef MEM_BOUNDARY
  #define MEM_BOUNDARY ((int) 32)
#endif

void print_array(const int col_length,
                 const int row_length,
                 const bool is_row_major,
                 const double * arr)
{
  int i, j;
  if (is_row_major)
  {
    for (i = 0; i < col_length; i++) {
      for (j = 0; j < row_length; j++)
      {
        printf("%f ", arr[i * col_length + j]);
      }

      printf("\n");
    }
  } else
  {
    for (i = 0; i < col_length; i++) {
      for (j = 0; j < row_length; j++)
      {
        printf("%f ", arr[j * col_length + i]);
      }

      printf("\n");
    }
  }

  printf("\n");
}

inline void transpose_array(const int col_length,
                            const int row_length,
                            const double * restrict A,
                                  double * restrict B)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);

  int row, col;

  for (col = 0; col < col_length; col++)
  {
    for (row = 0; row < row_length; row++)
    {
      B[(row * row_length) + col] = A[(col * col_length) + row];
    }
  }
}

/**
 * Multiplies a 4 x P matrix by a P x 4 matrix using AVX instructions.
 * Requires that all matrix inputs be aligned to 32 bytes or shit hits the fan *
 * @method dgemm_4P4
 * @param  A         4 x P matrix in row-order
 * @param  B         P x 4 matrix in column-order
 * @param  C         4 x 4 matrix in column-order
 */
inline void dgemm_4P4(const double * restrict A,
                      const double * restrict B,
                            double * restrict C,
                            double * restrict A_transposed)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);
  __assume_aligned(C, MEM_BOUNDARY);
  __assume_aligned(A_transposed, MEM_BOUNDARY);

  transpose_array(4, MUL_BLOCK_SIZE, A, A_transposed);

  __m256d C_1 = _mm256_load_pd(C + 0 * MUL_BLOCK_SIZE);
  __m256d C_2 = _mm256_load_pd(C + 1 * MUL_BLOCK_SIZE);
  __m256d C_3 = _mm256_load_pd(C + 2 * MUL_BLOCK_SIZE);
  __m256d C_4 = _mm256_load_pd(C + 3 * MUL_BLOCK_SIZE);

  int i;
  for (i = 0; i < MUL_BLOCK_SIZE; i++)
  {
    __m256d A_col = _mm256_load_pd(A_transposed + MUL_BLOCK_SIZE * i);

    __m256d B_1 = _mm256_broadcast_sd(B + MUL_BLOCK_SIZE * 0 + i);
    __m256d B_2 = _mm256_broadcast_sd(B + MUL_BLOCK_SIZE * 1 + i);
    __m256d B_3 = _mm256_broadcast_sd(B + MUL_BLOCK_SIZE * 2 + i);
    __m256d B_4 = _mm256_broadcast_sd(B + MUL_BLOCK_SIZE * 3 + i);

    // Calculate dot products
    __m256d DP_1 = _mm256_mul_pd(A_col, B_1);
    __m256d DP_2 = _mm256_mul_pd(A_col, B_2);
    __m256d DP_3 = _mm256_mul_pd(A_col, B_3);
    __m256d DP_4 = _mm256_mul_pd(A_col, B_4);

    C_1 = _mm256_add_pd(C_1, DP_1);
    C_2 = _mm256_add_pd(C_2, DP_2);
    C_3 = _mm256_add_pd(C_3, DP_3);
    C_4 = _mm256_add_pd(C_4, DP_4);
  }

  _mm256_store_pd(C + 0 * MUL_BLOCK_SIZE, C_1);
  _mm256_store_pd(C + 1 * MUL_BLOCK_SIZE, C_2);
  _mm256_store_pd(C + 2 * MUL_BLOCK_SIZE, C_3);
  _mm256_store_pd(C + 3 * MUL_BLOCK_SIZE, C_4);
}

inline void dgemm_8x8(const double * restrict A,
                      const double * restrict B,
                            double * restrict C,
                            double * restrict scratch_4P4)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);
  __assume_aligned(C, MEM_BOUNDARY);
  __assume_aligned(scratch_4P4, MEM_BOUNDARY);

  int i;
  for (i = 0; i < 8; i++)
  {
    dgemm_4P4(A, B, C, scratch_4P4);
    dgemm_4P4(A + 32, B, C + 16, scratch_4P4);
    dgemm_4P4(A, B + 32, C + 32, scratch_4P4);
    dgemm_4P4(A + 32, B + 32, C + 48, scratch_4P4);
  }
}

// inline void dgemm_64x64(const double * restrict A,
//                         const double * restrict B,
//                               double * restrict C,
//                               double * restrict scratch_4P4)
// {
//   __assume_aligned(A, MEM_BOUNDARY);
//   __assume_aligned(B, MEM_BOUNDARY);
//   __assume_aligned(C, MEM_BOUNDARY);
//   __assume_aligned(scratch_4P4, MEM_BOUNDARY);
//
//   const int block_numel = 4 * MUL_BLOCK_SIZE;
//   int i;
//
//   for (i = 0; i < MUL_BLOCK_SIZE; i++) {
//     dgemm_4P4(A + 0  * block_numel, B + i * block_numel, C + 0  * 16, scratch_4P4);
//     dgemm_4P4(A + 1  * block_numel, B + i * block_numel, C + 1  * 16, scratch_4P4);
//     dgemm_4P4(A + 2  * block_numel, B + i * block_numel, C + 2  * 16, scratch_4P4);
//     dgemm_4P4(A + 3  * block_numel, B + i * block_numel, C + 3  * 16, scratch_4P4);
//     dgemm_4P4(A + 4  * block_numel, B + i * block_numel, C + 4  * 16, scratch_4P4);
//     dgemm_4P4(A + 5  * block_numel, B + i * block_numel, C + 5  * 16, scratch_4P4);
//     dgemm_4P4(A + 6  * block_numel, B + i * block_numel, C + 6  * 16, scratch_4P4);
//     dgemm_4P4(A + 7  * block_numel, B + i * block_numel, C + 7  * 16, scratch_4P4);
//     dgemm_4P4(A + 8  * block_numel, B + i * block_numel, C + 8  * 16, scratch_4P4);
//     dgemm_4P4(A + 9  * block_numel, B + i * block_numel, C + 9  * 16, scratch_4P4);
//     dgemm_4P4(A + 10 * block_numel, B + i * block_numel, C + 10 * 16, scratch_4P4);
//     dgemm_4P4(A + 11 * block_numel, B + i * block_numel, C + 11 * 16, scratch_4P4);
//     dgemm_4P4(A + 12 * block_numel, B + i * block_numel, C + 12 * 16, scratch_4P4);
//     dgemm_4P4(A + 13 * block_numel, B + i * block_numel, C + 13 * 16, scratch_4P4);
//     dgemm_4P4(A + 14 * block_numel, B + i * block_numel, C + 14 * 16, scratch_4P4);
//     dgemm_4P4(A + 15 * block_numel, B + i * block_numel, C + 15 * 16, scratch_4P4);
//   }
// }

void square_dgemm(const int     M,
                  const double *A,
                  const double *B,
                        double *C)
{
    const int malloc_block_size = MUL_BLOCK_SIZE * MUL_BLOCK_SIZE * sizeof(double);
    const int n_blocks = M / MUL_BLOCK_SIZE + (M % MUL_BLOCK_SIZE? 1 : 0);

    print_array(M, M, false, B);

    int blocked_row, blocked_col, blocked_k;
    int row, col, row_start, col_start, row_size, col_size;
    int k, k_start, k_size;

    // Preallocate holding space for the blocks
    double * a_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double * b_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double * c_block = _mm_malloc(malloc_block_size, MEM_BOUNDARY);

    // Preallocate scratch arrays
    double * scratch_4P4 = _mm_malloc(4 * MUL_BLOCK_SIZE * sizeof(double), MEM_BOUNDARY);

    // Create a copy of A that is transposed in row-major order
    // to get element at (col,row): a_transposed[row * M + col]
    double * a_transposed = _mm_malloc(M * M * sizeof(double), MEM_BOUNDARY);
    transpose_array(M, M, A, a_transposed);

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

            for (k = 0; k < k_size; k++)
            {
              // a_block[(k * MUL_BLOCK_SIZE) + row] = a_transposed[row_start_inner + k];
              a_block[(row * MUL_BLOCK_SIZE) + k] = a_transposed[row_start_inner + k];
            }
          }

          // Copying B into a contiguous block of memory
          for (col = 0; col < col_size; col++)
          {
          	int col_start_inner = ((col_start + col)*M) + k_start;

            for (k = 0; k < k_size; k++)
            {
              b_block[(col * MUL_BLOCK_SIZE) + k] =  B[col_start_inner + k];
            }
          }

          // printf("A:\n");
          // print_array(MUL_BLOCK_SIZE, MUL_BLOCK_SIZE, true, a_block);

          // printf("B:\n");
          print_array(MUL_BLOCK_SIZE, MUL_BLOCK_SIZE, false, b_block);

          // Write into temporary array C assuming A is row-major and B is column-major
          // dgemm_4P4(a_block, b_block, c_block, scratch_4P4);
          dgemm_8x8(a_block, b_block, c_block, scratch_4P4);
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
    _mm_free(a_transposed);
    _mm_free(scratch_4P4);
    _mm_free(c_block);
    _mm_free(b_block);
    _mm_free(a_block);
}
