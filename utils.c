#include <immintrin.h>

// #define ALIGN __attribute__ ((aligned (32)))

/**
 * [dgemm_4x4 description]
 * @method dgemm_4x4
 * @param  A         4 x 4 matrix in row-order
 * @param  B         4 x 4 matrix in column-order
 * @param  C         4 x 4 matrix in column-order
 */
inline void dgemm_4x4(const double * restrict A, const double * restrict B, double * restrict C)
{
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  // Extract all rows from A
  __m256d row1 = _mm256_load_pd(A);
  __m256d row2 = _mm256_load_pd(A+4);
  __m256d row3 = _mm256_load_pd(A+8);
  __m256d row4 = _mm256_load_pd(A+12);

  // Extract all columns from B
  __m256d col1 = _mm256_load_pd(B);
  __m256d col2 = _mm256_load_pd(B+4);
  __m256d col3 = _mm256_load_pd(B+8);
  __m256d col4 = _mm256_load_pd(B+12);

  // Evaluate C[:,1]
  __m256 row1_col1 = _mm256_mul_pd(row1, col1);
  __m256 row2_col1 = _mm256_mul_pd(row2, col1);
  __m256 row3_col1 = _mm256_mul_pd(row3, col1);
  __m256 row4_col1 = _mm256_mul_pd(row4, col1);

  __m256 result_col1 = _mm256_add_pd(row1_col1, row2_col1);
  result_col1 = _mm256_add_pd(result_col1, row3_col1);
  result_col1 = _mm256_add_pd(result_col1, row4_col1);

  // Evaluate C[:,2]
  __m256 row1_col2 = _mm256_mul_pd(row1, col2);
  __m256 row2_col2 = _mm256_mul_pd(row2, col2);
  __m256 row3_col2 = _mm256_mul_pd(row3, col2);
  __m256 row4_col2 = _mm256_mul_pd(row4, col2);

  __m256 result_col2 = _mm256_add_pd(row1_col2, row2_col2);
  result_col2 = _mm256_add_pd(result_col2, row3_col2);
  result_col2 = _mm256_add_pd(result_col2, row4_col2);

  // Evaluate C[:,3]
  __m256 row1_col3 = _mm256_mul_pd(row1, col3);
  __m256 row2_col3 = _mm256_mul_pd(row2, col3);
  __m256 row3_col3 = _mm256_mul_pd(row3, col3);
  __m256 row4_col3 = _mm256_mul_pd(row4, col3);

  __m256 result_col3 = _mm256_add_pd(row1_col3, row2_col3);
  result_col3 = _mm256_add_pd(result_col3, row3_col3);
  result_col3 = _mm256_add_pd(result_col3, row4_col3);

  // Evaluate C[:,4]
  __m256 row1_col4 = _mm256_mul_pd(row1, col4);
  __m256 row2_col4 = _mm256_mul_pd(row2, col4);
  __m256 row3_col4 = _mm256_mul_pd(row3, col4);
  __m256 row4_col4 = _mm256_mul_pd(row4, col4);

  __m256 result_col4 = _mm256_add_pd(row1_col4, row2_col4);
  result_col4 = _mm256_add_pd(result_col4, row3_col4);
  result_col4 = _mm256_add_pd(result_col4, row4_col4);

  // Writeback to the result
  _mm_store_pd(C, result_col_1);
  _mm_store_pd(C+4, result_col_2);
  _mm_store_pd(C+8, result_col_3);
  _mm_store_pd(C+12, result_col_4);
}
