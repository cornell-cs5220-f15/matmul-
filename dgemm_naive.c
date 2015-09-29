#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char* dgemm_desc = "naive copy-blocked dgemm.";

#ifndef BLOCK_SIZE
  #define BLOCK_SIZE ((int) 4)
#endif

#ifndef MEM_BOUNDARY
  #define MEM_BOUNDARY ((int) 32)
#endif

void transpose_array(const int M, const double *A, double *copied)
{
  int row, column;

  for (column = 0; column < M; ++column) {
    for (row = 0; row < M; ++row) {
      copied[(row * M) + column] = A[(column * M) + row];
    }
  }
}

// loops over A and B and writes to C
// assumes that A is row major and B is column major
void stupid_dgemm(const int M, double *A, double *B, double *C)
{
  int col, row, k;

  // write to C in row-major order
  for (col = 0; col < M; ++col) {
    for (row = 0; row < M; ++row) {
      double cij = C[col * M + row];

      int a_start = row * M;
      int b_start = col * M;

      for (k = 0; k < M; ++k) {
        cij += A[a_start + k] * B[b_start + k];
      }

      C[col*M+row] = cij;
    }
  }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE ? 1 : 0);
  int blocked_row, blocked_col, blocked_k, row_start, col_start, k_start, row, col, k, col_size, row_size, k_size;

  // create A copy of A that is transposed in row-major order
  // to get element at (col,row): a_transposed[row*M + col]
  double *a_transposed = (double*) _mm_malloc(M * M * sizeof(double), MEM_BOUNDARY);
  transpose_array(M, A, a_transposed);

  int malloc_block_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);

  double* a_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);
  double* b_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);
  double* c_block = (double*) _mm_malloc(malloc_block_size, MEM_BOUNDARY);

  for (blocked_col = 0; blocked_col < n_blocks; ++blocked_col) {
    col_start = blocked_col * BLOCK_SIZE;
    col_size =  ((col_start + BLOCK_SIZE) > M) ? (M - col_start) : BLOCK_SIZE;

    for (blocked_row = 0; blocked_row < n_blocks; ++blocked_row) {
      row_start = blocked_row * BLOCK_SIZE;
      row_size = ((row_start + BLOCK_SIZE) > M) ? (M - row_start) : BLOCK_SIZE;

      // reset temporary matrices to zero
      c_block = memset(c_block, 0, malloc_block_size);

      for (blocked_k = 0; blocked_k < n_blocks; ++blocked_k) {
        // reset temporary matrices to zero
        a_block = memset(a_block, 0, malloc_block_size);
        b_block = memset(b_block, 0, malloc_block_size);

        // C = A[blocked_k, blocked_row] * B[blocked_col, blocked_k]
        k_start = blocked_k * BLOCK_SIZE;
        k_size = ((k_start + BLOCK_SIZE) > M) ? (M - k_start) : BLOCK_SIZE;

        // copying A into a contiguous block of memory
        for (row = 0; row < row_size; ++row) {
          int row_start_inner = ((row_start + row) * M) + k_start;

          for (k = 0; k < k_size; ++k) {
            a_block[(row * BLOCK_SIZE) + k] = a_transposed[row_start_inner + k];
          }
        }

        // copying B into a contiguous block of memory
        for (col = 0; col < col_size; ++col) {
          int col_start_inner = ((col_start + col)*M) + k_start;
          for (k = 0; k < k_size; ++k) {
            b_block[(col * BLOCK_SIZE) + k] =  B[col_start_inner + k];
          }
        }

        // do stupid_dgemm assuming a is row_major and b is column major
        // writes into temporary array C
        stupid_dgemm(BLOCK_SIZE, a_block, b_block, c_block);
      }

      // write c_block into actual location in C
      for (col = 0; col < col_size; ++col) {
        for (row = 0; row < row_size; ++row) {
          C[((col_start + col) * M) + row_start + row] = c_block[col*BLOCK_SIZE + row];
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
