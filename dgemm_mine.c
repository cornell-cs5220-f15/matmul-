const char* dgemm_desc = "My awesome dgemm.";

// #include <nmmintrin.h>
#include <immintrin.h>

// Block size that is used to fit submatrices into L1 cache
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

// Block size that is used to fit submatrices into register
#ifndef SUB_BLOCK_SIZE
#define SUB_BLOCK_SIZE ((int) 8)
#endif

#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B,
                 double* restrict C)//, const double restrict *C_original)
{

    // New kernal function for A stored in row-major.
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
    // My kernal function that uses AVX for optimal performance
    // It always assumes an input of A = 2 * P, B = P * 2
    // Template from https://bitbucket.org/dbindel/cs5220-s14/wiki/sse

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

void scatter_vec(double* addr, __m256d data) {
  // Extract lower 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d lower_data = _mm256_extractf128_pd(data, 0);
  _mm_storel_pd(addr+0, lower_data);
  _mm_storeh_pd(addr+1, lower_data);

  // Extract higher 128b from SIMD register and store each double to
  // appropriate memory location.
  __m128d higher_data = _mm256_extractf128_pd(data, 1);
  _mm_storel_pd(addr+2, higher_data);
  _mm_storeh_pd(addr+3, higher_data);
}

void mine_fma_dgemm( const double* restrict A, const double* restrict B,
                 double* restrict C){
    // My kernal function that utilizes the architecture of totient node.
    // It uses the 256 bits register size which accomodate 4 doubles
    // Also, it tries to use FMA to maximize the computational efficiency.

    // To do this, it assumes an input of A = 4*4 and B = 4*4 with output C = 4*4
    // The size can be changed later for better performance, but 4*4 will be a good choice for prototyping
    // The matrices are all assumed to be stored in column major

    const int Matrix_size = 4;

    // A command that I got from S14 code. Helps compiler optimize (?not too sure)
    // Should be 32 here
    __assume_aligned(A, 32);
    __assume_aligned(B, 32);
    __assume_aligned(C, 32);

    // Load matrix A
    __m256d a0 = _mm256_load_pd(A + Matrix_size * 0);
    __m256d a1 = _mm256_load_pd(A + Matrix_size * 1);
    __m256d a2 = _mm256_load_pd(A + Matrix_size * 2);
    __m256d a3 = _mm256_load_pd(A + Matrix_size * 3);

    scatter_vec(C, a1);
    // int it, jt;
    //
    // printf("============Matrix C inside the loop ============\n");
    //  for (it = 0; it < 4; ++it ){
    //   for (jt = 0; jt < 4; ++jt){
    //     printf("%f\t", C[jt*4+it]);
    //   }
    //   printf("\n");
    // }


    // Load matrix C
    __m256d c0 = _mm256_load_pd(C + Matrix_size * 0);
    __m256d c1 = _mm256_load_pd(C + Matrix_size * 1);
    __m256d c2 = _mm256_load_pd(C + Matrix_size * 2);
    __m256d c3 = _mm256_load_pd(C + Matrix_size * 3);

    // Preallocate one vector for entries of b
    __m256d bij;
    // Core routine to update C using FMA
    int i;
    for (i = 0; i < Matrix_size; i++) {
      bij = _mm256_set1_pd(*(B+i*Matrix_size));
      c0  = _mm256_fmadd_pd(a0, bij, c0); // C = A * B + C;
      bij = _mm256_set1_pd(*(B+i*Matrix_size+1));
      c1  = _mm256_fmadd_pd(a1, bij, c1); // C = A * B + C;
      bij = _mm256_set1_pd(*(B+i*Matrix_size+2));
      c2  = _mm256_fmadd_pd(a2, bij, c2); // C = A * B + C;
      bij = _mm256_set1_pd(*(B+i*Matrix_size+3));
      c3  = _mm256_fmadd_pd(a3, bij, c3); // C = A * B + C;
    }

    // _mm256_storeu_pd ((double *) C, a1);
    // scatter_vec(C, bij);
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

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    // Preallocate a space for submatrices A, B and C
    // double* A_transposed = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // double* A_transposed = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),16);
    // double* B_transposed = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),16);

    // Assign blocks for kernals to perform fast computation.
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0); // # of blocks
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi){
      const int i = bi * BLOCK_SIZE;
      for (bk = 0; bk < n_blocks; ++bk){
        const int k = bk * BLOCK_SIZE;

        // Transpose A. This part needs to be rewritten for clarity and performance
        const int M_sub = (i+BLOCK_SIZE > M? M-i : BLOCK_SIZE);
        const int K = (k+BLOCK_SIZE > M? M-k : BLOCK_SIZE);

        // int it, kt;
        // for (it = 0; it < M_sub; ++it){
        //   for (kt = 0; kt < K; ++kt){
        //     A_transposed[it*BLOCK_SIZE + kt] = A[i + k*M + it + kt*M];
        //   }
        // }

        // Instead of transposing A, transpose B for AVX 2*2
        // for (it = 0; it < M_sub; ++it){
        //   for (kt = 0; kt < K; ++kt){
        //     B_transposed[it*BLOCK_SIZE + kt] = B[i + k*M + it + kt*M];
        //   }
        // }

        // Don't transpose anything for AVX 4*4
        for (bj = 0; bj < n_blocks; ++bj){
          const int j = bj * BLOCK_SIZE;
          // int it, jt;
          // printf("Matrix B_transposed\n");
          // for (it = 0; it < M; ++it ){
          //   for (jt = 0; jt < M; ++jt){
          //     printf("%f\t", B_transposed[jt*BLOCK_SIZE+it]);
          //   }
          //   printf("\n");
          // }

          // do_block(M, A_transposed, B, C, i, j, k);
          // do_block(M, A, B_transposed, C, i, j, k); // For AVX 2*2
          do_block(M, A, B, C, i, j, k); // For AVX 4*4
        }

      }
    }

    int it, jt;
    printf("Matrix A\n");
    for (it = 0; it < M; ++it ){
      for (jt = 0; jt < M; ++jt){
        printf("%f\t", A[jt*M+it]);
      }
      printf("\n");
    }

    printf("Matrix B\n");
    for (it = 0; it < M; ++it ){
      for (jt = 0; jt < M; ++jt){
        printf("%f\t", B[jt*M+it]);
      }
      printf("\n");
    }

    printf("Matrix C\n");
     for (it = 0; it < M; ++it ){
      for (jt = 0; jt < M; ++jt){
        printf("%f\t", C[jt*M+it]);
      }
      printf("\n");
    }

    // _mm_free(A_transposed);
    // _mm_free(B_transposed);
}
