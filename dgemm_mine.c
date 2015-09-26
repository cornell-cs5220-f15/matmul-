const char* dgemm_desc = "My awesome dgemm.";

// #include <nmmintrin.h>
#include <immintrin.h>

// Block size that is used to fit submatrices into L1 cache
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

// Block size that is used to fit submatrices into register
#ifndef INNER_BLOCK_SIZE
#define INNER_BLOCK_SIZE ((int) 4)
#endif

#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif


void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B,
                 double* restrict C)
{
  /* For the new kernal function, A must be stored in row-major
    lda is the leading dimension of the matrix (the M of square_dgemm).
    A is M-by-K, B is K-by-N and C is M-by-N */
    int i, j, k;
    for(j = 0; j < N; ++j){
      for(i = 0; i < M; ++i){
        double cij = C[i + j*lda];
        for (k = 0; k < K; ++k){
          cij += A[i * BLOCK_SIZE + k] * B[k + j * lda];
        }
        C[i + j*lda] = cij;
      }
    }
}

void mine_dgemm( const double* restrict A, const double* restrict B,
                 double* restrict C){
    /*  My kernal function that uses AVX for optimal performance
        It always assumes an input of A = 2 * P, B = P * 2
        Template from https://bitbucket.org/dbindel/cs5220-s14/wiki/sse
        B should be stored in its row form.
    */

    double C_swap = C[1];
    C[1] = C[3];
    C[3] = C_swap;

    // This is really implicit in using the aligned ops...
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    // Load diagonal and off-diagonals
    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    /*
    Assuming 2*2 case, and we do traditional, naive three loop multiplication.
    Except in this case we don't need any loop because it's small.
    */

    __m128d a0 = _mm_load_pd(A);
    __m128d b0 = _mm_load_pd(B);
    __m128d td0 = _mm_mul_pd(a0, b0);
    __m128d bs0 = swap_sse_doubles(b0);
    __m128d to0 = _mm_mul_pd(a0, bs0);

    __m128d a1 = _mm_load_pd(A+2);
    __m128d b1 = _mm_load_pd(B+BLOCK_SIZE);
    __m128d td1 = _mm_mul_pd(a1, b1);
    __m128d bs1 = swap_sse_doubles(b1);
    __m128d to1 = _mm_mul_pd(a1, bs1);

    __m128d td_sum = _mm_add_pd(td0, td1);
    __m128d to_sum = _mm_add_pd(to0, to1);

    cd = _mm_add_pd(cd, td_sum);
    co = _mm_add_pd(co, to_sum);

    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);

    C_swap = C[3];
    C[3] = C[1];
    C[1] = C_swap;
}

void mine_fma_dgemm( const double* restrict A, const double* restrict B,
                 double* restrict C){
    /*
    My kernal function that utilizes the architecture of totient node.
    It uses the 256 bits register size which accomodate 4 doubles
    Also, it uses FMA to maximize the computational efficiency.
    To do this, it assumes an input of A = 4*4 and B = 4*4 with output C = 4*4
    The size can be changed later for better performance, but 4*4 will be a good choice for prototyping
    The matrices are all assumed to be stored in column major
     */

    const int Matrix_size = 4;

    // A command that I got from S14 code. Helps compiler optimize
    __assume_aligned(A, 32);
    __assume_aligned(B, 32);
    __assume_aligned(C, 32);

    // Load matrix A. The load function is loadu, which doesn't require alignment of the meory
    __m256d a0 = _mm256_loadu_pd((A + Matrix_size * 0));
    __m256d a1 = _mm256_loadu_pd((A + Matrix_size * 1));
    __m256d a2 = _mm256_loadu_pd((A + Matrix_size * 2));
    __m256d a3 = _mm256_loadu_pd((A + Matrix_size * 3));

    __m256d bij;
    int i, j;
    for (i = 0; i < Matrix_size; i++){
      // Load one column of C, C(:,i)
      __m256d c = _mm256_loadu_pd((C + Matrix_size*i));
      // Perform FMA on A*B(:,i)
      bij = _mm256_set1_pd(*(B+i*Matrix_size+0));
      c = _mm256_fmadd_pd(a0, bij, c);
      bij = _mm256_set1_pd(*(B+i*Matrix_size+1));
      c = _mm256_fmadd_pd(a1, bij, c);
      bij = _mm256_set1_pd(*(B+i*Matrix_size+2));
      c = _mm256_fmadd_pd(a2, bij, c);
      bij = _mm256_set1_pd(*(B+i*Matrix_size+3));
      c = _mm256_fmadd_pd(a3, bij, c);
      // Store C(:,i)
      _mm256_storeu_pd((C+i*Matrix_size),c);
    }
}


void do_block(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k)
{
    // Determine the size of each sub-block
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    // basic_dgemm(lda, M, N, K, A, B + k + j*lda, C + i + j*lda);
    // mine_dgemm(A,B,C);
    mine_fma_dgemm(A,B,C);
}

void matrix_copy (const int mat_size, const int sub_size, const int i, const int j,
        const double* restrict Matrix, double* restrict subMatrix){
  // Get a copy of submatrix
  const int M = (i+sub_size > mat_size? mat_size-i : sub_size); // Maybe we can do this outside, but I'm not worried about this right now.
  const int N = (j+sub_size > mat_size? mat_size-j : sub_size);
  // Make a copy
  int m, n;
  for (m = 0; m < M; m++){
    for (n = 0; n < N; n++){
      subMatrix[m*sub_size + n] = Matrix[(i+m)*sub_size + (j+n)];
    }
  }
  // Populate the submatrix with 0 to enforce regular pattern in the computation.
  for (m = 0; m < (sub_size - M); m++){
    for (n = 0; n < (sub_size - N); n++){
      subMatrix[m*sub_size + n] = 0.0;
    }
  }
}
void matrix_update (const int mat_size, const int sub_size, const int i, const int j,
        double* restrict Matrix, const double* restrict subMatrix){
    int m, n;
    const int M = (i+sub_size > mat_size? mat_size-i : sub_size);
    const int N = (j+sub_size > mat_size? mat_size-j : sub_size);
    for (m = 0; m < M; m++){
      for (n = 0; n < N; n++){
         Matrix[(i+m)*sub_size + (j+n)] = subMatrix[m*sub_size + n];
      }
    }
}


void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    // // Preallocate a space for submatrices A, B and C
    // double* A_transposed = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // double* A_transposed = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),16);
    // double* B_transposed = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),16);

    // Preallocate spaces for outer matrices A, B and C;
    double* A_outer = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),32);
    double* B_outer = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),32);
    double* C_outer = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),32);
    // Preallocate spaces for inner matrices A, B and C;
    double* A_inner = (double*) _mm_malloc(INNER_BLOCK_SIZE * INNER_BLOCK_SIZE * sizeof(double),32);
    double* B_inner = (double*) _mm_malloc(INNER_BLOCK_SIZE * INNER_BLOCK_SIZE * sizeof(double),32);
    double* C_inner = (double*) _mm_malloc(INNER_BLOCK_SIZE * INNER_BLOCK_SIZE * sizeof(double),32);

    // Testing code
    const int n_inner_blocks = M / INNER_BLOCK_SIZE + (M%INNER_BLOCK_SIZE? 1 : 0); // # of blocks
    int sbi, sbj, sbk;
    for (sbi = 0; sbi < n_inner_blocks; sbi++){
      for (sbj = 0; sbj < n_inner_blocks; sbj++){
        matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbj, C, C_inner);
        for (sbk = 0; sbk < n_inner_blocks; sbk++){
          matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbk, A, A_inner);
          matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbk, sbj, B, B_inner);
          mine_fma_dgemm(A_inner, B_inner, C_inner);
          int it, jt;
          printf("Super Inside, Matrix C_inner is:\n");
          for(it = 0; it < M; it ++){
            for(jt = 0; jt < M; jt ++){
              printf("%lf \t", C_inner[it*M+jt]);
            }
            printf("\n");
          }
        }
        matrix_update (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbj, C, C_inner);
        int it, jt;
        printf("Inside, Matrix C_inner is:\n");
        for(it = 0; it < M; it ++){
          for(jt = 0; jt < M; jt ++){
            printf("%lf \t", C_inner[it*M+jt]);
          }
          printf("\n");
        }
      }
    }
    int it, jt;
    printf("Outside , Matrix C is: \n");
    for(it = 0; it < M; it ++){
      for(jt = 0; jt < M; jt ++){
        printf("%lf \t", C[it*M+jt]);
      }
      printf("\n");
    }
    // // Assign blocks for kernals to perform fast computation.
    // const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0); // # of blocks
    // const int n_inner_blocks = BLOCK_SIZE / INNER_BLOCK_SIZE; // For convenience, choose block size to be multiple of inner block size.
    // int bi, bj, bk;
    // int sbi, sbj, sbk;
    // for (bi = 0; bi < n_blocks; bi++){
    //   for (bj = 0; bj < n_blocks; bj++){
    //     matrix_copy (M, BLOCK_SIZE, bi, bj, C, C_outer);
    //     for (bk = 0; bk < n_blocks; bk++){
    //       matrix_copy (M, BLOCK_SIZE, bi, bk, A, A_outer);
    //       matrix_copy (M, BLOCK_SIZE, bk, bj, B, B_outer);
    //       // Compute the block multiplication by passing submatrices to kernal function
    //       for (sbi = 0; sbi < n_inner_blocks; sbi++){
    //         for (sbj = 0; sbj < n_inner_blocks; sbj++){
    //           matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbj, C_outer, C_inner);
    //           for (sbk = 0; sbk < n_inner_blocks; sbk++){
    //             matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbk, A_outer, A_inner);
    //             matrix_copy (BLOCK_SIZE, INNER_BLOCK_SIZE, sbk, sbj, B_outer, B_inner);
    //             mine_fma_dgemm(A_inner, B_inner, C_inner);
    //           }
    //           matrix_update (BLOCK_SIZE, INNER_BLOCK_SIZE, sbi, sbj, C_outer, C_inner);
    //         }
    //       }
    //     }
    //     matrix_update (M, BLOCK_SIZE, bi, bj, C, C_outer);
    //   }
    // }

    // int it, jt;
    // printf("Matrix A is:");
    // for(it = 0; it < M; it ++){
    //   for(jt = 0; jt < M; jt ++){
    //     printf("%lf \t", A[it*M+jt]);
    //   }
    //   printf("\n");
    // }
    // printf("Matrix B is:");
    // for(it = 0; it < M; it ++){
    //   for(jt = 0; jt < M; jt ++){
    //     printf("%lf \t", B[it*M+jt]);
    //   }
    //   printf("\n");
    // }
    // printf("Matrix C is:");
    // for(it = 0; it < M; it ++){
    //   for(jt = 0; jt < M; jt ++){
    //     printf("%lf \t", C[it*M+jt]);
    //   }
    //   printf("\n");
    // }


    // int bi, bj, bk;
    // for (bi = 0; bi < n_blocks; ++bi){
    //   const int i = bi * BLOCK_SIZE;
    //   for (bk = 0; bk < n_blocks; ++bk){
    //     const int k = bk * BLOCK_SIZE;
    //     for (bj = 0; bj < n_blocks; ++bj){
    //       const int j = bj * BLOCK_SIZE;
    //       // do_block(M, A_transposed, B, C, i, j, k);
    //       // do_block(M, A, B_transposed, C, i, j, k); // For AVX 2*2
    //       do_block(M, A, B, C, i, j, k); // For AVX 4*4
    //     }
    //     // // Transpose A.
    //     // const int M_sub = (i+BLOCK_SIZE > M? M-i : BLOCK_SIZE);
    //     // const int K = (k+BLOCK_SIZE > M? M-k : BLOCK_SIZE);
    //     // int it, kt;
    //
    //     // for (it = 0; it < M_sub; ++it){
    //     //   for (kt = 0; kt < K; ++kt)
    //     //     A_transposed[it*BLOCK_SIZE + kt] = A[i + k*M + it + kt*M];
    //     // }
    //
    //     // // Instead of transposing A, transpose B for AVX 2*2
    //     // for (it = 0; it < M_sub; ++it){
    //     //   for (kt = 0; kt < K; ++kt)
    //     //     B_transposed[it*BLOCK_SIZE + kt] = B[i + k*M + it + kt*M];
    //     // }
    //
    //     // // Don't transpose anything for AVX 4*4
    //     // for (bj = 0; bj < n_blocks; ++bj){
    //     //   const int j = bj * BLOCK_SIZE;
    //     //   // do_block(M, A_transposed, B, C, i, j, k);
    //     //   // do_block(M, A, B_transposed, C, i, j, k); // For AVX 2*2
    //     //   do_block(M, A, B, C, i, j, k); // For AVX 4*4
    //     // }
    //   }
    // }
    // // Free memory for basic kernal and AVX kernal.
    // _mm_free(A_transposed);
    // _mm_free(B_transposed);
}
