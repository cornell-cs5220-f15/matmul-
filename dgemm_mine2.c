

const char* dgemm_desc = "My 3 level blocked dgemm.";

#include <immintrin.h>
#include <x86intrin.h>

#ifndef L3_BLOCK_SIZE
#define L3_BLOCK_SIZE ((int) 400)
#define L3 L3_BLOCK_SIZE
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE ((int) 40)
#define L2 L2_BLOCK_SIZE
#endif

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE ((int) 4)
#define L1 L1_BLOCK_SIZE
#endif

//L0 fits into registers, we have a fast kernel for that part
//right now the same as L1 blocks. This many blocking levels tend to slow
//the computation (don't know exactly why.)
#ifndef L0_BLOCK_SIZE
#define L0_BLOCK_SIZE ((int) 4)
#define L0 L0_BLOCK_SIZE
#endif

#define N2 L3/L2
#define N1 L2/L1
#define N0 L1/L0


#include <stdlib.h>
#include <stdio.h>

//Fast kernel for 4x4 row major matrix multiplication.
//(It was for column major layout at the beginning, C = A*B assuming col major,
// but I switched the order of the arguments
// so now it reads C' = B' * A' if you think column major layout.)

void MMult4by4VRegAC(const double* restrict B, const double* restrict A, double* restrict C)
{
  __m256d a0, a1;
  __m256d b0, b1, b2, b3, b4, b5, b6, b7;
  __m256d c0,c1,c2,c3;
  c0 = _mm256_load_pd(C+0);
  c1 = _mm256_load_pd(C+4);
  c2 = _mm256_load_pd(C+8);
  c3 = _mm256_load_pd(C+12);

{
    a0 = _mm256_load_pd(A);
    b0 = _mm256_broadcast_sd(B);
    b1 = _mm256_broadcast_sd(B+4);
    b2 = _mm256_broadcast_sd(B+8);
    b3 = _mm256_broadcast_sd(B+12);

    c0 = _mm256_fmadd_pd(a0,b0,c0);
    c1 = _mm256_fmadd_pd(a0,b1,c1);
    c2 = _mm256_fmadd_pd(a0,b2,c2);
    c3 = _mm256_fmadd_pd(a0,b3,c3);

    a1 = _mm256_load_pd(A+4);
    b4 = _mm256_broadcast_sd(B+1);
    b5 = _mm256_broadcast_sd(B+5);
    b6 = _mm256_broadcast_sd(B+9);
    b7 = _mm256_broadcast_sd(B+13);

    c0 = _mm256_fmadd_pd(a1,b4,c0);
    c1 = _mm256_fmadd_pd(a1,b5,c1);
    c2 = _mm256_fmadd_pd(a1,b6,c2);
    c3 = _mm256_fmadd_pd(a1,b7,c3);

    a0 = _mm256_load_pd(A+8);
    b0 = _mm256_broadcast_sd(B+2);
    b1 = _mm256_broadcast_sd(B+6);
    b2 = _mm256_broadcast_sd(B+10);
    b3 = _mm256_broadcast_sd(B+14);

    c0 = _mm256_fmadd_pd(a0,b0,c0);
    c1 = _mm256_fmadd_pd(a0,b1,c1);
    c2 = _mm256_fmadd_pd(a0,b2,c2);
    c3 = _mm256_fmadd_pd(a0,b3,c3);

    a1 = _mm256_load_pd(A+12);
    b4 = _mm256_broadcast_sd(B+3);
    b5 = _mm256_broadcast_sd(B+7);
    b6 = _mm256_broadcast_sd(B+11);
    b7 = _mm256_broadcast_sd(B+15);

    c0 = _mm256_fmadd_pd(a1,b4,c0);
    c1 = _mm256_fmadd_pd(a1,b5,c1);
    c2 = _mm256_fmadd_pd(a1,b6,c2);
    c3 = _mm256_fmadd_pd(a1,b7,c3);
  }
  _mm256_store_pd(C,c0);
  _mm256_store_pd(C+4,c1);
  _mm256_store_pd(C+8,c2);
  _mm256_store_pd(C+12,c3);
}



// read (L1 x L1) matrix from A(i,j) into block_A (assumes col major A)
// block_A is row major (historical reasons)
void read_to_contiguous(const int M, const double* restrict A, double* restrict block_A,
                        const int i, const int j) {
    // guard against matrix edge case
    const int mBound = (j+L1 > M? M-j : L1);
    const int nBound = (i+L1 > M? M-i : L1);

    // offset is index of upper left corner of desired block within A
    const int offset = i + M*j;
    int m, n;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            block_A[m + L1*n] = A[offset + m*M + n];
        }
        while (m < L1){
            block_A[m + L1*n] = 0.0;
            m++;
        }
    }
    while (n < L1) {
        for (m = 0; m < L1; m++)
            block_A[m + L1*n] = 0.0;
        n++;
    }
}

// write block_C into C(i,j)
void write_from_contiguousC(const int M, double* restrict C,
                            const double* restrict block_C,
                            const int i, const int j) {
    // guard against matrix edge case
    // printf("%d %d\n", i, j);
    const int mBound = (i+L1 > M? M-i : L1); // rows
    const int nBound = (j+L1 > M? M-j : L1); // cols

    int m, n;
    const int offset = i + M * j;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            C[offset + m + M * n] = block_C[m * L1 + n];
        }
    }
}

//Assumes L3 is integer mutliple of L2 and L2 is integer multiple of L1
void to_contiguous3lvlBlock(const int M,
                            const double* restrict A,
                            double* restrict Ak,
                            const double* restrict B,
                            double* restrict Bk) {
    int ind_Ak = 0, ind_Bk = 0;
    const int n3 = M / L3_BLOCK_SIZE + (M%L3_BLOCK_SIZE? 1 : 0);
    for(int i = 0; i < n3; i++){
        const int row_i = i * L3;
        const int n2_i = (i == n3-1 && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
        for (int j = 0; j < n3; j++){
            const int col_j = j * L3;
            const int n2_j = (j == n3-1 && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
            for(int q = 0; q < n2_i; q++){
                const int row_q = row_i + q * L2;
                const int n1_q = (i == n3-1 && q == n2_i-1 && (M%L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                for (int s = 0; s < n2_j; s++){
                    const int col_s = col_j + s * L2;
                    const int n1_s = (j == n3-1 && s == n2_j-1 && (M%L2)? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                    for (int m = 0; m < n1_q; m++){
                        for (int n = 0; n < n1_s; n++){
                            read_to_contiguous(M, A, Ak + ind_Ak, row_q + m * L1, col_s + n * L1);
                            ind_Ak += L1*L1;
                            read_to_contiguous(M, B, Bk + ind_Bk, col_s + n * L1, row_q + m * L1);
                            ind_Bk += L1*L1;
                        }
                    }
                }
            }
        }

    }
}

void from_contiguous3lvlBlock(const int M,
                              double* restrict C,
                              const double* restrict Ck){
    int ind_Ck = 0;
    const int n3 = M / L3 + (M%L3? 1 : 0);
    for(int i = 0; i < n3; i++){
        const int row_i = i * L3;
        const int n2_i = (i == n3-1  && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
        for (int j = 0; j < n3; j++){
            const int col_j = j * L3;
            const int n2_j = (j == n3-1  && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
            for(int q = 0; q < n2_i; q++){
                const int row_q = row_i + q * L2;
                const int n1_q = (i == n3-1 && q == n2_i-1  && (M % L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                for (int s = 0; s < n2_j; s++){
                    const int col_s = col_j + s * L2;
                    const int n1_s = (j == n3-1 && s == n2_j - 1  && (M % L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                    for (int m = 0; m < n1_q; m++){
                        for (int n = 0; n < n1_s; n++){
                            write_from_contiguousC(M, C, Ck + ind_Ck, row_q + m * L1, col_s + n * L1);
                            ind_Ck += L1*L1;
                        }
                    }
                }
            }
        }

    }
}

// void do_block_L1(const double* restrict Ak, const double* restrict Bk, double* restrict Ck) {
//     //Each block is L1xL1, we padded the original matrix with zeros
//     //to make sure this is the case
//     // basic_square_dgemm(L1, Ak, Bk, Ck);
//     // return;
//     int bi, bj, bk;
//     int i, j, jB, k;
//     for (bi = 0; bi < N0; bi ++) {
//         i = bi * L0 * L1;
//         for (bj = 0; bj < N0; bj++) {
//             jB = bj * L0 * L1;
//             j = bj * L0 * L0;
//             const int ind_Ck = i + j;
//             for (bk = 0; bk < N0; bk++) {
//                 MMult4by4VRegAC(Ak + i + bk*L0*L0, Bk + j + bk*L0*L0, Ck + ind_Ck);
//             }
//         }
//     }
//
// }
void do_block_L2(const double* restrict Ak, const double* restrict Bk, double* restrict Ck,
                  const int M, const int N, const int K) {

    int bi, bj, bk;
    const int ni = M / L1 + (M%L1? 1 : 0); // number of blocks in M rows
    const int nj = N / L1 + (N%L1? 1 : 0); // number of blocks in N cols
    const int nk = K / L1 + (K%L1? 1 : 0); // number of blocks in K rows

    int iA, iC, j, jB, k;
    int MM, NN, KK;

    int sizeOfBlock_C = L1 * L1;
    int sizeOfBlock_B = L1 * L1;
    for (bi = 0; bi < ni-1; bi++) {
        iA = bi * K * L1;
        iC = bi * N * L1;
        for (bj = 0; bj < nj-1; bj++) {
            j = bj * sizeOfBlock_C;
            jB = bj * K * L1;
            const int ind_Ck = iC+j;
            for (bk = 0; bk < nk; bk++) {
                const int ind_Ak = iA + sizeOfBlock_C * bk;
                const int ind_Bk = jB + sizeOfBlock_B * bk;
                MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
            }
        }
        j = bj * sizeOfBlock_C;
        jB = bj * K * L1;
        sizeOfBlock_B = ( N%L1 ? L1 * (N%L1) : L1 * L1);
        const int ind_Ck = iC+j;
        for (bk = 0; bk < nk; bk++) {
            const int ind_Ak = iA + sizeOfBlock_C * bk;
            const int ind_Bk = jB + sizeOfBlock_B * bk;
            MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
        }
    }
    sizeOfBlock_C = M%L1 ? L1 * (M%L1) : L1 * L1;
    iA = bi * K * L1;
    iC = bi * N * L1;
    for (bj = 0; bj < nj-1; bj++) {
        j = bj * sizeOfBlock_C;
        jB = bj * K * L1;
        const int ind_Ck = iC+j;
        for (bk = 0; bk < nk; bk++) {
            const int ind_Ak = iA + sizeOfBlock_C * bk;
            const int ind_Bk = jB + sizeOfBlock_B * bk;
            MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
        }
    }
    j = bj * sizeOfBlock_C;
    jB = bj * K * L1;
    sizeOfBlock_B = ( N%L1 ? L1 * (N%L1) : L1 * L1);
    const int ind_Ck = iC+j;
    for (bk = 0; bk < nk; bk++) {
        const int ind_Ak = iA + sizeOfBlock_C * bk;
        const int ind_Bk = jB + sizeOfBlock_B * bk;
        MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
    }

}
// inline void do_block_L3(const double* restrict Ak, const double* restrict Bk, double* restrict Ck,
//                   const int M, const int N, const int K) {
//     __attribute__((always_inline));
//     int bi, bj, bk;
//     const int ni = M / L2 + (M%L2? 1 : 0); // number of blocks in M rows
//     const int nj = N / L2 + (N%L2? 1 : 0); // number of blocks in N cols
//     const int nk = K / L2 + (K%L2? 1 : 0); // number of blocks in K rows
//
//     int ind_Ak, ind_Bk, ind_Ck;
//     int iA, iC, j, jB, k;
//     int MM, NN, KK;
//     MM = L2;
//     NN = L2;
//     int sizeOfBlock_C = L2 * L2;
//     int sizeOfBlock_B = L2 * L2;
//     for (bi = 0; bi < ni-1; bi++) {
//         iA = bi * K * L2;
//         iC = bi * N * L2;
//         sizeOfBlock_B = L2 * L2;
//         for (bj = 0; bj < nj-1; bj++) {
//             j = bj * sizeOfBlock_C;
//             jB = bj * K * L2;
//             ind_Ck = iC+j;
//             for (bk = 0; bk < nk-1; bk++) {
//                 ind_Ak = iA + sizeOfBlock_C * bk;
//                 ind_Bk = jB + sizeOfBlock_B * bk;
//                     //multiply blocks: C_ij += A_ik * B_kj
//                     //A is MM by KK
//                     //B is KK by NN
//                     //C is MM by NN
//                 do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, L2, L2, L2);
//             }
//             ind_Ak = iA + sizeOfBlock_C * bk;
//             ind_Bk = jB + sizeOfBlock_B * bk;
//             KK = (K%L2) ? K%L2 : L2;
//             do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, L2, L2, KK);
//         }
//         NN = N%L2 ? N%L2 : L2;
//         sizeOfBlock_B = ( bj == nj-1 && N%L2 ? L2 * (N%L2) : L2 * L2);
//         j = bj * sizeOfBlock_C;
//         jB = bj * K * L2;
//         ind_Ck = iC+j;
//         for (bk = 0; bk < nk-1; bk++) {
//             ind_Ak = iA + sizeOfBlock_C * bk;
//             ind_Bk = jB + sizeOfBlock_B * bk;
//             do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, L2, NN, L2);
//         }
//         ind_Ak = iA + sizeOfBlock_C * bk;
//         ind_Bk = jB + sizeOfBlock_B * bk;
//         KK = (K%L2) ? K%L2 : L2;
//         do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, L2, NN, KK);
//     }
//     sizeOfBlock_C = M%L2 ? L2 * (M%L2) : L2 * L2;
//     sizeOfBlock_B = L2*L2;
//     MM = M%L2 ? M%L2 : L2;
//     iA = bi * K * L2;
//     iC = bi * N * L2;
//     for (bj = 0; bj < nj-1; bj++) {
//         j = bj * sizeOfBlock_C;
//         jB = bj * K * L2;
//         ind_Ck = iC+j;
//         for (bk = 0; bk < nk-1; bk++) {
//             ind_Ak = iA + sizeOfBlock_C * bk;
//             ind_Bk = jB + sizeOfBlock_B * bk;
//                 //multiply blocks: C_ij += A_ik * B_kj
//                 //A is MM by KK
//                 //B is KK by NN
//                 //C is MM by NN
//             do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, L2, L2);
//         }
//         ind_Ak = iA + sizeOfBlock_C * bk;
//         ind_Bk = jB + sizeOfBlock_B * bk;
//         KK = (K%L2) ? K%L2 : L2;
//         do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, L2, KK);
//     }
//     NN = N%L2 ? N%L2 : L2;
//     sizeOfBlock_B = (N%L2 ? L2 * (N%L2) : L2 * L2);
//     j = bj * sizeOfBlock_C;
//     jB = bj * K * L2;
//     ind_Ck = iC+j;
//     for (bk = 0; bk < nk-1; bk++) {
//         ind_Ak = iA + sizeOfBlock_C * bk;
//         ind_Bk = jB + sizeOfBlock_B * bk;
//         do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, L2);
//     }
//     ind_Ak = iA + sizeOfBlock_C * bk;
//     ind_Bk = jB + sizeOfBlock_B * bk;
//     KK = (K%L2) ? K%L2 : L2;
//     do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, KK);
// }
void do_block_L3(const double* restrict Ak, const double* restrict Bk, double* restrict Ck,
                  const int M, const int N, const int K) {

    int bi, bj, bk;
    const int ni = M / L2 + (M%L2? 1 : 0); // number of blocks in M rows
    const int nj = N / L2 + (N%L2? 1 : 0); // number of blocks in N cols
    const int nk = K / L2 + (K%L2? 1 : 0); // number of blocks in K rows

    int ind_Ak, ind_Bk, ind_Ck;
    int iA, iC, j, jB, k;
    int MM, NN, KK;

    for (bi = 0; bi < ni; bi++) {
        iA = bi * K * L2;
        iC = bi * N * L2;
        int sizeOfBlock_C = ( bi == ni-1 && M%L2 ? L2 * (M%L2) : L2 * L2);
        MM = ( bi == ni-1 && M%L2) ? M%L2 : L2;
        for (bj = 0; bj < nj; bj++) {
            NN = ( bj == nj-1 && N%L2) ? N%L2 : L2;
            j = bj * sizeOfBlock_C;
            jB = bj * K * L2;
            int sizeOfBlock_B = ( bj == nj-1 && N%L2 ? L2 * (N%L2) : L2 * L2);
            const int ind_Ck = iC+j;
            for (bk = 0; bk < nk; bk++) {
                const int ind_Ak = iA + sizeOfBlock_C * bk;
                const int ind_Bk = jB + sizeOfBlock_B * bk;
                KK = (bk == nk-1 && K%L2) ? K%L2 : L2;
                    //multiply blocks: C_ij += A_ik * B_kj
                    //A is MM by KK
                    //B is KK by NN
                    //C is MM by NN
                    // printf("\t%d %d %d\n", ind_Ak, ind_Bk, ind_Ck);
                    // printf("\t%d %d %d\n",ni, nj, nk);

                do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, KK);
            }
        }
    }

}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C) {
    const int N = (M / L1 + (M%L1? 1 : 0)) * L1; // new size after padding with zeros

//Begin copying to new layout
    double* Ak = _mm_malloc(N*N*sizeof(double), 32);
    double* Bk = _mm_malloc(N*N*sizeof(double), 32);
    double* Ck = _mm_malloc(N*N*sizeof(double), 32);

    for (int i = 0; i<N*N; i++)
        Ck[i] = 0.0;
    to_contiguous3lvlBlock(M, A, Ak, B, Bk);
//End of copying

    const int n3 = N / L3 + (N%L3? 1 : 0); //Number of L3 block in one dimension

    int i, j, k;
    int MM, NN, KK; //The sizes of the blocks (at the edges they may be rectangular)

    //C_ij = A_ik * B_kj
    for (int bi = 0; bi < n3; bi++){
        i = bi * N * L3;
        int sizeOfBlock_C = ( bi == n3-1 && N%L3 ? L3 * (N%L3) : L3 * L3);
        MM = ( bi == n3-1 && N%L3) ? N%L3 : L3;
        for (int bj = 0; bj < n3; bj++){
            NN = ( bj == n3-1 && N%L3) ? N%L3 : L3;
            j = bj * sizeOfBlock_C;
            int jB = bj * N * L3;
            int sizeOfBlock_B = ( bj == n3-1 && N%L3 ? L3 * (N%L3) : L3 * L3);
            const int ind_Ck = i+j;
            for (int bk = 0; bk < n3; bk++){
                int ind_Ak = i + sizeOfBlock_C * bk;
                int ind_Bk = jB + sizeOfBlock_B * bk;
                KK = (bk == n3-1 && N%L3) ? N%L3 : L3;

                do_block_L3(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, KK);
            }
        }
    }
    _mm_free(Ak);
    _mm_free(Bk);
    from_contiguous3lvlBlock(M, C, Ck);
    _mm_free(Ck);
}
