const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK3_SIZE
#define BLOCK3_SIZE ((int) 65)
#endif

#ifndef BLOCK2_SIZE
#define BLOCK2_SIZE ((int) 130)
#endif

#ifndef BLOCK1_SIZE
#define BLOCK1_SIZE ((int) 260)
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
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK3_SIZE > lda? lda-i : BLOCK3_SIZE);
    const int N = (j+BLOCK3_SIZE > lda? lda-j : BLOCK3_SIZE);
    const int K = (k+BLOCK3_SIZE > lda? lda-k : BLOCK3_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void do_block2(const int lda,
              const double * A, const double * B, double * C,
              const int i, const int j, const int k)
{
    const int n_blocks = BLOCK2_SIZE / BLOCK3_SIZE + (BLOCK2_SIZE%BLOCK3_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int r = bi * BLOCK3_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int s = bj * BLOCK3_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int t = bk * BLOCK3_SIZE;
                do_block(lda, A, B, C, i+r, j+s, k+t);
            }
        }
    }
}

void do_block1(const int lda,
              const double * A, const double * B, double * C,
              const int i, const int j, const int k)
{
    const int n_blocks = BLOCK1_SIZE / BLOCK2_SIZE + (BLOCK1_SIZE%BLOCK2_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int r = bi * BLOCK2_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int s = bj * BLOCK2_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int t = bk * BLOCK2_SIZE;
                do_block2(lda, A, B, C, i+r, j+s, k+t);
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK1_SIZE + (M%BLOCK1_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK1_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK1_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK1_SIZE;
                do_block1(M, A, B, C, i, j, k);
            }
        }
    }
}

