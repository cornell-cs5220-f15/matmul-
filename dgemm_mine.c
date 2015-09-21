const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *At, const double *B, double *C, const double *C_original)
{
    // int i, j, k;
    // for (i = 0; i < M; ++i) {
    //     for (j = 0; j < N; ++j) {
    //         double cij = C[j*lda+i];
    //         for (k = 0; k < K; ++k) {
    //             cij += A[k*lda+i] * B[j*lda+k];
    //         }
    //         C[j*lda+i] = cij;
    //     }
    // }

    // New kernal for A stored in row-major.
    int i, j, k;


    for(j = 0; j < N; ++j){
      for(i = 0; i < M; ++i){
        double cij = C[i + j*lda];
        for (k = 0; k < K; ++k){
          cij += At[ i * BLOCK_SIZE + k ] * B[k + j * lda ];
        }
        C[i + j*lda] = cij;
        // printf("%f\n", C[i + j*lda]);
      }
      // printf("**********One subroutine\n" );
    }
    // printf("==========One routine\n" );
    // printf("lda is %d\n", lda);
    int pmm, pnn;
    // printf("\nMatrix C:\n");
    for (pmm = 0; pmm < lda; ++pmm){
      for (pnn = 0; pnn < lda; ++pnn){
        // printf("%f\t", C_original[pmm + pnn*lda]);
      }
      // printf("\n");
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C, double *At,
              const int i, const int j, const int k)
{
    // Determine the size of each sub-block
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

    basic_dgemm(lda, M, N, K, At, B + k + j*lda, C + i + j*lda, C);
                /*A + i + k*lda*/
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Preallocate a space for matrix A
    double* A_transposed = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // Assign blocks for kernals to perform fast computation.
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0); // # of blocks
    int pmm, pnn;
    // printf("\nMatrix A:\n");
    // for (pmm = 0; pmm < M; ++pmm){
    //   printf("%f\t",A[pmm]);
    // }
    // printf("\n");
    // for (pmm = 0; pmm < M; ++pmm){
    //   for (pnn = 0; pnn < M; ++pnn){
        // printf("%f\t", A[pmm + pnn*M]);
      // }
      // printf("\n");
    // }
    // printf("\nMatrix B:\n");
    // for (pmm = 0; pmm < M; ++pmm){
    //   for (pnn = 0; pnn < M; ++pnn){
    //     printf("%f\t", B[pmm + pnn*M]);
    //   }
    //   printf("\n");
    // }

    int bi, bj, bk;
    // int counter = 0;
    for (bi = 0; bi < n_blocks; ++bi){
      const int i = bi * BLOCK_SIZE;
      for (bk = 0; bk < n_blocks; ++bk){
        const int k = bk * BLOCK_SIZE;

        // Transpose A. This part needs to be rewritten for clarity and performance
        const int A_start = i + k*M;
        const int M_sub = (i+BLOCK_SIZE > M? M-i : BLOCK_SIZE);
        const int K = (k+BLOCK_SIZE > M? M-k : BLOCK_SIZE);
        int it, kt;
        // printf("A Start is %d\n", A_start);
        for (it = 0; it < M_sub; ++it){
          for (kt = 0; kt < K; ++kt){
            A_transposed[it*BLOCK_SIZE + kt] = A[A_start + it + kt*M];
          }
        }

        // printf("A Transpose is\n");
        // for (pmm = 0; pmm < BLOCK_SIZE; ++pmm){
        //   for (pnn = 0; pnn < BLOCK_SIZE; ++pnn){
        //     printf("%f\t", A_transposed[pmm*BLOCK_SIZE + pnn]);
        //   }
        //   printf("\n");
        // }
        //
        // for (pmm = 0; pmm < BLOCK_SIZE*BLOCK_SIZE; ++pmm){
        //     printf("%f\t", A_transposed[pmm]);
        // }
        // printf("\n");

        for (bj = 0; bj < n_blocks; ++bj){
          const int j = bj * BLOCK_SIZE;
          // printf("Round: %d\n", ++counter);
          do_block(M, A, B, C, A_transposed, i, j, k);
        }
      }
    }
    free(A_transposed);
}
