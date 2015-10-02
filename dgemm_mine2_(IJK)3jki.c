const char* dgemm_desc = "My 3-level-blocked dgemm with order (I-J-K)*3-j-k-i.";

#ifndef BLOCK_SIZE1
#define BLOCK_SIZE1 ((int) 48)
#endif
#ifndef BLOCK_SIZE2
#define BLOCK_SIZE2 ((int) 96)
#endif
#ifndef BLOCK_SIZE3
#define BLOCK_SIZE3 ((int) 800)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    double cj[BLOCK_SIZE1];
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i){
            cj[i] = C[j*lda+i];
        }
        for (k = 0; k < K; ++k) {
            for (i = 0; i < M; ++i) {
                cj[i] += A[k*lda+i] * B[j*lda+k];
            }
        }
        for (i = 0; i < M; ++i){
            C[j*lda+i] = cj[i];
        }
    }
    //printf("%.0f, %.0f, %.0f\n", A[0], B[0], C[0]);
    /*
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k)
            printf("%.0f ", A[k*lda+i]);
        printf("\n");
    }
    printf("\n");
    for (k = 0; k < K; ++k) {
        for (j = 0; j < N; ++j)
            printf("%.0f ", B[j*lda+k]);
        printf("\n");
    }
    printf("*****************\n\n");
     */
}

void do_block1(const int lda, const int MM, const int NN, const int KK,
               const double *A, const double *B, double *C,
               const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE1 > MM? MM-i : BLOCK_SIZE1);
    const int N = (j+BLOCK_SIZE1 > NN? NN-j : BLOCK_SIZE1);
    const int K = (k+BLOCK_SIZE1 > KK? KK-k : BLOCK_SIZE1);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}



void dgemm2(const int lda, const int M, const int N, const int K,
            const double *A, const double *B, double *C)
{
    const int m_blocks = M / BLOCK_SIZE1 + (M%BLOCK_SIZE1? 1 : 0);
    const int n_blocks = N / BLOCK_SIZE1 + (N%BLOCK_SIZE1? 1 : 0);
    const int k_blocks = K / BLOCK_SIZE1 + (K%BLOCK_SIZE1? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < m_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE1;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE1;
            for (bk = 0; bk < k_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE1;
                //printf("%.0f, %.0f, %.0f\n", A[0], B[0], C[0]);
                do_block1(lda, M, N, K, A, B, C, i, j, k);
            }
        }
    }
}

void do_block2(const int lda, const int MM, const int NN, const int KK,
               const double *A, const double *B, double *C,
               const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE2 > MM? MM-i : BLOCK_SIZE2);
    const int N = (j+BLOCK_SIZE2 > NN? NN-j : BLOCK_SIZE2);
    const int K = (k+BLOCK_SIZE2 > KK? KK-k : BLOCK_SIZE2);
    //printf("%d, %d %d %d, %.0f %.0f %.0f\n",lda, M,N,K, A[i + k*lda], B[k + j*lda], C[i + j*lda]);
    dgemm2(lda, M, N, K,
           A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void dgemm3(const int lda, const int M, const int N, const int K,
            const double *A, const double *B, double *C)
{
    const int m_blocks = M / BLOCK_SIZE2 + (M%BLOCK_SIZE2? 1 : 0);
    const int n_blocks = N / BLOCK_SIZE2 + (N%BLOCK_SIZE2? 1 : 0);
    const int k_blocks = K / BLOCK_SIZE2 + (K%BLOCK_SIZE2? 1 : 0);
    //printf("%d %d %d\n", m_blocks, n_blocks, k_blocks);
    int bi, bj, bk;
    for (bi = 0; bi < m_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE2;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE2;
            for (bk = 0; bk < k_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE2;
                //printf("%.0f, %.0f, %.0f\n", A[0], B[0], C[0]);
                do_block2(lda, M, N, K, A, B, C, i, j, k);
            }
        }
    }
}

    void do_block3(const int lda,
                   const double *A, const double *B, double *C,
                   const int i, const int j, const int k)
    {
        const int M = (i+BLOCK_SIZE3 > lda? lda-i : BLOCK_SIZE3);
        const int N = (j+BLOCK_SIZE3 > lda? lda-j : BLOCK_SIZE3);
        const int K = (k+BLOCK_SIZE3 > lda? lda-k : BLOCK_SIZE3);
        //printf("%d, %d %d %d, %.0f %.0f %.0f\n",lda, M,N,K, A[i + k*lda], B[k + j*lda], C[i + j*lda]);
        dgemm3(lda, M, N, K,
               A + i + k*lda, B + k + j*lda, C + i + j*lda);
    }
    
    void square_dgemm(const int M, const double *A, const double *B, double *C)
    {
        const int n_blocks = M / BLOCK_SIZE3 + (M%BLOCK_SIZE3? 1 : 0);
        int bi, bj, bk;
        for (bi = 0; bi < n_blocks; ++bi) {
            const int i = bi * BLOCK_SIZE3;
            for (bj = 0; bj < n_blocks; ++bj) {
                const int j = bj * BLOCK_SIZE3;
                for (bk = 0; bk < n_blocks; ++bk) {
                    const int k = bk * BLOCK_SIZE3;
                    do_block3(M, A, B, C, i, j, k);
                }
            }
        }
    }

