const char* dgemm_desc = "My awesome dgemm.";

#include <nmmintrin.h>

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

    int P = 2;
    // This is really implicit in using the aligned ops...
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    // Load diagonal and off-diagonals
    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    /*
     * Do block dot product.  Each iteration adds the result of a two-by-two
     * matrix multiply into the accumulated 2-by-2 product matrix, which is
     * stored in the registers cd (diagonal part) and co (off-diagonal part).
     */
    for (int k = 0; k < P; k += 2) {

        __m128d a0 = _mm_load_pd(A+2*k+0);
        __m128d b0 = _mm_load_pd(B+2*k+0);
        __m128d td0 = _mm_mul_pd(a0, b0);
        __m128d bs0 = swap_sse_doubles(b0);
        __m128d to0 = _mm_mul_pd(a0, bs0);

        __m128d a1 = _mm_load_pd(A+2*k+2);
        __m128d b1 = _mm_load_pd(B+2*k+2);
        __m128d td1 = _mm_mul_pd(a1, b1);
        __m128d bs1 = swap_sse_doubles(b1);
        __m128d to1 = _mm_mul_pd(a1, bs1);

        __m128d td_sum = _mm_add_pd(td0, td1);
        __m128d to_sum = _mm_add_pd(to0, to1);

        cd = _mm_add_pd(cd, td_sum);
        co = _mm_add_pd(co, to_sum);
    }

    // Write back sum
    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);
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
    mine_dgemm(A,B,C);
}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    // Preallocate a space for submatrices A, B and C
    // double* A_transposed = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double* A_transposed = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double),16);
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
        int it, kt;
        for (it = 0; it < M_sub; ++it){
          for (kt = 0; kt < K; ++kt){
            A_transposed[it*BLOCK_SIZE + kt] = A[i + k*M + it + kt*M];
          }
        }

        for (bj = 0; bj < n_blocks; ++bj){
          const int j = bj * BLOCK_SIZE;
          do_block(M, A_transposed, B, C, i, j, k);
        }

      }
    }

    int it, jt;
    for (it = 0; it < M; ++it ){
      for (jt = 0; jt < M; ++jt){
        printf("%f\t", &C[jt*M+it]);
      }
      printf("\n");
    }
    _mm_free(A_transposed);
}
