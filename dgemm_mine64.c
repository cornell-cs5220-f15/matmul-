#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const char* dgemm_desc = "Copy optimized, blocked dgemm using an AVX kernel";

#ifndef INS_SIZE
  #define INS_SIZE ((int) 4)
#endif

#ifndef MEM_BOUNDARY
  #define MEM_BOUNDARY ((int) 32)
#endif

void print_array(int num_rows,
                 int num_cols,
                 bool is_row_major,
                 const double * arr)
{
  int i, j;
  if (is_row_major)
  {
    for (i = 0; i < num_rows; i++) {
      for (j = 0; j < num_cols; j++)
      {
        // printf("%02d, %f ", i * num_cols + j, arr[i * num_cols + j]);
        printf("%f ", arr[i * num_cols + j]);
      }

      printf("\n");
    }
  } else
  {
    for (i = 0; i < num_rows; i++) {
      for (j = 0; j < num_cols; j++)
      {
        // printf("%02d, %f ", j * num_rows + i, arr[j * num_rows + i]);
        printf("%f ", arr[j * num_rows + i]);
      }

      printf("\n");
    }
  }

  printf("\n");
}

// inline void transpose_array(const int num_rows,
//                             const int num_cols,
//                             const double * restrict A,
//                                   double * restrict B)
// {
//   __assume_aligned(A, MEM_BOUNDARY);
//   __assume_aligned(B, MEM_BOUNDARY);
//
//   int row, col;
//
//   for (col = 0; col < num_rows; col++)
//   {
//     for (row = 0; row < num_cols; row++)
//     {
//       B[(row * num_rows) + col] = A[(col * num_rows) + row];
//     }
//   }
// }

/**
 * Multiplies a 4 x P matrix by a P x 4 matrix using AVX instructions.
 * Requires that all matrix inputs be aligned to 32 bytes or shit hits the fan *
 * @method dgemm_4P4
 * @param  A         4 x P matrix in row-order
 * @param  B         P x 4 matrix in column-order
 * @param  C         4 x 4 matrix in column-order
 */
inline void dgemm_4P4(const int M,
                      const int row_offset,
                      const int col_offset,
                      const int block_numel,
                      const double * restrict A,
                      const double * restrict B,
                            double * restrict C)
{
  __assume_aligned(A, MEM_BOUNDARY);
  __assume_aligned(B, MEM_BOUNDARY);
  __assume_aligned(C, MEM_BOUNDARY);

  const int offset = (col_offset * INS_SIZE) + (row_offset * block_numel);

  __m256d C_1 = _mm256_load_pd(C + offset + (0 * M));
  __m256d C_2 = _mm256_load_pd(C + offset + (1 * M));
  __m256d C_3 = _mm256_load_pd(C + offset + (2 * M));
  __m256d C_4 = _mm256_load_pd(C + offset + (3 * M));

  int i;
  for (i = 0; i < M; i++)
  {
    __m256d A_col = _mm256_load_pd(A + INS_SIZE * i);

    __m256d B_1 = _mm256_broadcast_sd(B + M * 0 + i);
    __m256d B_2 = _mm256_broadcast_sd(B + M * 1 + i);
    __m256d B_3 = _mm256_broadcast_sd(B + M * 2 + i);
    __m256d B_4 = _mm256_broadcast_sd(B + M * 3 + i);

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

  _mm256_store_pd(C + offset + (0 * M), C_1);
  _mm256_store_pd(C + offset + (1 * M), C_2);
  _mm256_store_pd(C + offset + (2 * M), C_3);
  _mm256_store_pd(C + offset + (3 * M), C_4);

  print_array(M, M, false, C);
}

// inline void dgemm_8x8(const double * restrict A,
//                       const double * restrict B,
//                             double * restrict C,
//                             double * restrict scratch_4P4)
// {
//   __assume_aligned(A, MEM_BOUNDARY);
//   __assume_aligned(B, MEM_BOUNDARY);
//   __assume_aligned(C, MEM_BOUNDARY);
//   __assume_aligned(scratch_4P4, MEM_BOUNDARY);
//
//   int i;
//   for (i = 0; i < 8; i++)
//   {
//     dgemm_4P4(A, B, C, scratch_4P4);
//     dgemm_4P4(A + 32, B, C + 16, scratch_4P4);
//     dgemm_4P4(A, B + 32, C + 32, scratch_4P4);
//     dgemm_4P4(A + 32, B + 32, C + 48, scratch_4P4);
//   }
// }

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
//   const int block_numel = 4 * INS_SIZE;
//   int i;
//
//   for (i = 0; i < INS_SIZE; i++) {
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

void square_dgemm(const int      M,
                  const double * A,
                  const double * B,
                        double * C)
{
  setbuf(stdout, NULL);

  const int n_blocks = M / INS_SIZE + (M % INS_SIZE ? 1 : 0);

  const int block_numel = INS_SIZE * M;
  const int block_size = block_numel * sizeof(double);
  const int input_size = M * M * sizeof(double);

  // Preallocate caches for A and B
  double * a_aligned = _mm_malloc(input_size, MEM_BOUNDARY);
  double * b_aligned = _mm_malloc(input_size, MEM_BOUNDARY);

  // Preallocate holding space for the blocks
  double * a_block = _mm_malloc(block_size, MEM_BOUNDARY);
  double * b_block = _mm_malloc(block_size, MEM_BOUNDARY);
  double * c_block = _mm_malloc(block_size, MEM_BOUNDARY);

  // Copy A and B to their memory aligned substitues
  memcpy(a_aligned, A, input_size);
  memcpy(b_aligned, B, input_size);

  int i;
  // Iterate over the rows of A
  for (i = 0; i < n_blocks; i++)
  {
    // Copy out 4 rows of A into a_block in column-major.
    // Since A is in column major, we read 4 entries for the current block (i * INS_SIZE)
    // into a ymm register, then skip to the next column (j)
    int j;
    for (j = 0; j < block_numel; j+=INS_SIZE)
    {
      _mm256_store_pd(a_block + j, _mm256_load_pd(a_aligned + (i * INS_SIZE) + j));
    }

    // printf("A_block:\n");
    // print_array(4, M, false, a_block);
    // printf("\n");

    // Iterate over the columns of B
    for (j = 0; j < n_blocks; j++)
    {
      // Copy out 4 columns of B into b_block in column-major
      // Since B is in column major, we read 4 entries for the current block (j * block_numel)
      // into a ymm register in sequence.
      int k;
      for (k = 0; k < block_numel; k+=INS_SIZE)
      {
        _mm256_store_pd(b_block + k, _mm256_load_pd(b_aligned + (j * block_numel) + k));
      }

      // printf("B_block:\n");
      // print_array(M, 4, false, b_block);
      // printf("\n");

      // Multiply and update C
      dgemm_4P4(M, j, i, block_numel, a_block, b_block,  C);
    }
  }

  printf("A:\n");
  print_array(M, M, false, A);
  printf("\n");

  printf("B:\n");
  print_array(M, M, false, B);
  printf("\n");

  printf("C:\n");
  print_array(M, M, false, C);
  printf("\n");

  // Clean up in order of initialization
  _mm_free(c_block);
  _mm_free(b_block);
  _mm_free(a_block);

  _mm_free(b_aligned);
  _mm_free(a_aligned);
}
