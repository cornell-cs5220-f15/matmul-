

const char* dgemm_desc = "My 2 level blocked dgemm.";

#ifndef L3_BLOCK_SIZE
#define L3_BLOCK_SIZE ((int) 250)
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE ((int) 125)
#endif

#include <stdio.h>
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

// void do_block_L1(const int lda,
//               const double *A, const double *B, double *C,
//               const int i, const int j, const int k)
// {
//     const int M = (i+L1_BLOCK_SIZE > lda? lda-i : L1_BLOCK_SIZE);
//     const int N = (j+L1_BLOCK_SIZE > lda? lda-j : L1_BLOCK_SIZE);
//     const int K = (k+L1_BLOCK_SIZE > lda? lda-k : L1_BLOCK_SIZE);
//     basic_dgemm(lda, M, N, K,
//                 A + i + k*lda, B + k + j*lda, C + i + j*lda);
// }
//
// void do_block_L2(const int lda,
//               const double *A, const double *B, double *C,
//               const int ii, const int jj, const int kk)
// {
//
//     const int M = (i+L3_BLOCK_SIZE > lda? lda-i : L3_BLOCK_SIZE);
//     const int N = (j+L3_BLOCK_SIZE > lda? lda-j : L3_BLOCK_SIZE);
//     const int K = (k+L3_BLOCK_SIZE > lda? lda-k : L3_BLOCK_SIZE);
//
//     const int nL2_row_blocks_A = M / L2_BLOCK_SIZE + (M%L2_BLOCK_SIZE? 1 : 0);
//     const int nL2_col_blocks_A = K / L2_BLOCK_SIZE + (K%L2_BLOCK_SIZE? 1 : 0);
//     const int nL2_col_blocks_B = N / L2_BLOCK_SIZE + (N%L2_BLOCK_SIZE? 1 : 0);
//
//     int bi, bj, bk;
//     for (bi = 0; bi < nL2_row_blocks_A; ++bi) {
//         const int i = bi * L2_BLOCK_SIZE + ii;
//         for (bj = 0; bj < nL2_col_blocks_B; ++bj) {
//             const int j = bj * L2_BLOCK_SIZE + ii;
//             for (bk = 0; bk < nL2_col_blocks_A; ++bk) {
//                 const int k = bk * L2_BLOCK_SIZE + ii;
//                 do_block_L2(lda, A + ii + i + (kk+ k)*lda, B + kk + k + (jj+j)*lda, C + ii+ i +(jj + j)*lda);
//             }
//         }
//     }
// }

void do_block_L3(const int lda,
              const double *A, const double *B, double *C,
              const int ii, const int jj, const int kk)
{

    const int MM = (ii+L3_BLOCK_SIZE > lda? lda-ii : L3_BLOCK_SIZE);
    const int KK = (kk+L3_BLOCK_SIZE > lda? lda-kk : L3_BLOCK_SIZE);
    const int NN = (jj+L3_BLOCK_SIZE > lda? lda-jj : L3_BLOCK_SIZE);

    const int nL2_row_blocks_A = MM / L2_BLOCK_SIZE + (MM % L2_BLOCK_SIZE? 1 : 0);
    const int nL2_col_blocks_A = KK / L2_BLOCK_SIZE + (KK % L2_BLOCK_SIZE? 1 : 0);
    const int nL2_col_blocks_B = NN / L2_BLOCK_SIZE + (NN % L2_BLOCK_SIZE? 1 : 0);

    int bi, bj, bk;
    for (bi = 0; bi < nL2_row_blocks_A; ++bi) {
        const int i = bi * L2_BLOCK_SIZE + ii;
        const int M = (i+L2_BLOCK_SIZE > lda? lda-i : L2_BLOCK_SIZE);
        for (bj = 0; bj < nL2_col_blocks_B; ++bj) {
            const int j = bj * L2_BLOCK_SIZE + jj;
            const int N = (j+L2_BLOCK_SIZE > lda? lda-j : L2_BLOCK_SIZE);
            for (bk = 0; bk < nL2_col_blocks_A; ++bk) {
                const int k = bk * L2_BLOCK_SIZE + kk;
                const int K = (k+L2_BLOCK_SIZE > lda? lda-k : L2_BLOCK_SIZE);
                basic_dgemm(lda, M, N, K,
                            A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int nL3blocks = M / L3_BLOCK_SIZE + (M%L3_BLOCK_SIZE? 1 : 0);
    int b3i, b3j, b3k;
    for (b3i = 0; b3i < nL3blocks; ++b3i) {
        const int i = b3i * L3_BLOCK_SIZE;
        // printf("%d\t",i);
        for (b3j = 0; b3j < nL3blocks; ++b3j) {
            const int j = b3j * L3_BLOCK_SIZE;
            for (b3k = 0; b3k < nL3blocks; ++b3k) {
                const int k = b3k * L3_BLOCK_SIZE;
                do_block_L3(M, A, B, C, i, j, k);
                // printf("fuckfuckfuck\n" );
            }
        }
    }
}
