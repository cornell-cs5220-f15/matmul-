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

#ifndef BLOCK_SIZE
  #define BLOCK_SIZE ((int) 64)
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

  // print_array(M, M, false, C);
}

// assumes that A, B are column-major, 64x64 matrices
// DON'T NEED TO PAD W ZEROS. YAY.
void square_dgemm_64(const int M,
                     const double * A,
                     const double * B,
                     double * C,
                     double * a_block,
                     double * b_block,
                     const int n_blocks,
                     const int block_numel)
{
  int i, j;
  // Iterate over the rows of A
  for (i = 0; i < n_blocks; i++)
  {
    // Copy out 4 rows of A into a_block in column-major.
    // Since A is in column major, we read 4 entries for the current block (i * INS_SIZE)
    // into a ymm register, then skip to the next column (j)
    #pragma vector aligned
    for (j = 0; j < block_numel; j+=INS_SIZE)
    {
      _mm256_store_pd(a_block + j, _mm256_load_pd(A + (i * INS_SIZE) + (j / 4 * M)));
    }

    // printf("A_block:\n");
    // print_array(4, pad_dim, false, a_block);
    // printf("\n");

    // Iterate over the columns of B
    for (j = 0; j < n_blocks; j++)
    {
      // Copy out 4 columns of B into b_block in column-major
      // Since B is in column major, we read 4 entries for the current block (j * block_numel)
      // into a ymm register in sequence.
      int k;
      #pragma vector aligned
      for (k = 0; k < block_numel; k+=INS_SIZE)
      {
        _mm256_store_pd(b_block + k, _mm256_load_pd(B + (j * block_numel) + k));
      }

      // printf("B_block:\n");
      // print_array(pad_dim, 4, false, b_block);
      // printf("\n");

      // Multiply and update C
      dgemm_4P4(M, j, i, block_numel, a_block, b_block, C);
    }
  }

  // printf("C:\n");
  // print_array(M, M, false, C);
  // printf("\n");
}


void transpose_array(const int M, const double *A, double *copied)
{
    int row, column;

    for (column = 0; column < M; ++column){
        for (row = 0; row < M; ++row){
            copied[(row * M) + column] = A[(column * M) + row];
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int blocked_row, blocked_col, blocked_k, row_start, col_start, k_start, row, col, k, col_size, row_size, k_size;

    // create A copy of A that is transposed in row-major order
    // to get element at (col,row): a_transposed[row*M + col]
    double *a_transposed = (double*) _mm_malloc(M * M * sizeof(double), MEM_BOUNDARY);
    transpose_array(M, A, a_transposed);

    int malloc_block_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);

    double* a_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* b_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);
    double* c_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);

    // initializing inner function params

    const int n_blocks_inner = BLOCK_SIZE / INS_SIZE;
    const int input_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    const int block_numel = INS_SIZE * BLOCK_SIZE;
    const int block_size_inner = block_numel * sizeof(double);

    // Preallocate holding space for the blocks
    double * a_block_inner = _mm_malloc(block_size_inner, MEM_BOUNDARY);
    double * b_block_inner = _mm_malloc(block_size_inner, MEM_BOUNDARY);

    for (blocked_col = 0; blocked_col < n_blocks; ++blocked_col){
        col_start = blocked_col * BLOCK_SIZE;
        col_size =  ((col_start + BLOCK_SIZE) > M) ? (M - col_start) : BLOCK_SIZE;

        for (blocked_row = 0; blocked_row < n_blocks; ++blocked_row){
            row_start = blocked_row * BLOCK_SIZE;
            row_size = ((row_start + BLOCK_SIZE) > M) ? (M - row_start) : BLOCK_SIZE;

            // reset temporary matrices to zero
            c_block = memset(c_block, 0, malloc_block_size);

            for (blocked_k = 0; blocked_k < n_blocks; ++blocked_k){

                // reset temporary matrices to zero
                a_block = memset(a_block, 0, malloc_block_size);
                b_block = memset(b_block, 0, malloc_block_size);

                a_block_inner = memset(a_block_inner, 0, block_size_inner);
                b_block_inner = memset(b_block_inner, 0, block_size_inner);


                // C = A[blocked_k, blocked_row] * B[blocked_col, blocked_k]
                k_start = blocked_k * BLOCK_SIZE;
                k_size = ((k_start + BLOCK_SIZE) > M) ? (M - k_start) : BLOCK_SIZE;

                // copying A into a contiguous block of memory
                // in column major
                for (row = 0; row < row_size; ++row){
                  int row_start_inner = ((row_start + row) * M) + k_start;
                    for (k = 0; k < k_size; ++k){
                        a_block[(k * BLOCK_SIZE) + row] = a_transposed[row_start_inner + k];
                    }
                }

                // copying B into a contiguous block of memory
                // in column major
                for (col = 0; col < col_size; ++col){
                  int col_start_inner = ((col_start + col)*M) + k_start;
                    for (k = 0; k < k_size; ++k){
                        b_block[(col * BLOCK_SIZE) + k] =  B[col_start_inner + k];
                    }
                }

                square_dgemm_64(BLOCK_SIZE, a_block, b_block, c_block, a_block_inner, b_block_inner, n_blocks_inner, block_numel);
            }

            // write c_block into actual location in C
            for (col = 0; col < col_size; ++col){
                for (row = 0; row < row_size; ++row){
                    C[((col_start + col) * M) + row_start + row] = c_block[col*BLOCK_SIZE + row];
                }
            }
        }
    }

    _mm_free(a_transposed);
    _mm_free(a_block);
    _mm_free(b_block);
    _mm_free(c_block);
    _mm_free(b_block_inner);
    _mm_free(a_block_inner);
}
