const char* dgemm_desc = "My dgemm with copy optimization";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif
#ifndef ALIGNED_SIZE
#define ALIGNED_SIZE ((int) 64)
#endif


#define M 8
#define N 8
#define P 8

int DIM_M = M;
int DIM_N = N;
int DIM_K = P;


/*
 * On the Nehalem architecture, shufpd and multiplication use the same port.
 * 32-bit integer shuffle is a different matter.  If we want to try to make
 * it as easy as possible for the compiler to schedule multiplies along
 * with adds, it therefore makes sense to abuse the integer shuffle
 * instruction.  See also
 *   http://locklessinc.com/articles/interval_arithmetic/
 */
#define USE_SHUFPD
#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif


/*
 * Block matrix multiply kernel.
 * Inputs:
 *    A: 2-by-P matrix in column major format.
 *    B: P-by-2 matrix in row major format.
 * Outputs:
 *    C: 2-by-2 matrix with element order [c11, c22, c12, c21]
 *       (diagonals stored first, then off-diagonals)
 */
void kdgemm2P2(double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
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


/*
 * Block matrix multiply kernel.
 * Inputs:
 *    A: 4-by-P matrix in column major format.
 *    B: P-by-4 matrix in row major format.
 * Outputs:
 *    C: 4-by-4 matrix with element order 
 *       [c11, c22, c12, c21,   c31, c42, c32, c41,
 *        c13, c24, c14, c23,   c33, c44, c34, c43]
 *       That is, C is broken into 2-by-2 sub-blocks, and is stored
 *       in column-major order at the block level and diagonal/off-diagonal
 *       within blocks.
 */
void kdgemm4P4(double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    kdgemm2P2(C,    A+0,   B+0);
    kdgemm2P2(C+4,  A+2*P, B+0);
    kdgemm2P2(C+8,  A+0,   B+2*P);
    kdgemm2P2(C+12, A+2*P, B+2*P);
}

/*
 * Block matrix multiply kernel.
 * Inputs:
 *    A: 8-by-P matrix in column major format.
 *    B: P-by-8 matrix in row major format.
 * Outputs:
 *    C: 8-by-8 matrix viewed as a 2-by-2 block matrix.  Each block has
 *       the layout from kdgemm4P4.
 */
void kdgemm8P8(double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    kdgemm4P4(C,    A+0,   B+0);
    kdgemm4P4(C+16, A+4*P, B+0);
    kdgemm4P4(C+32, A+0,   B+4*P);
    kdgemm4P4(C+48, A+4*P, B+4*P);
}


/* 
 * Compute A to block row-major format, where each block is a 2-by-1 column.
 */
void to_kdgemm_A(int ldA,
                 const double * restrict A,
                 double * restrict Ak)
{
    for (int i = 0; i < M; i += 2)
        for (int j = 0; j < P; ++j) {
            Ak[0] = A[(i+0) + j*ldA];
            Ak[1] = A[(i+1) + j*ldA];
            Ak += 2;
        }
}


/* 
 * Compute B to block col-major format, where each block is a 1-by-2 row.
 */
void to_kdgemm_B(int ldB,
                 const double * restrict B,
                 double * restrict Bk)
{
    for (int j = 0; j < N; j += 2)
        for (int i = 0; i < P; ++i) {
            Bk[0] = B[i + (j+0)*ldB];
            Bk[1] = B[i + (j+1)*ldB];
            Bk += 2;
        }
}


/*
 * Convert a block from the kdgemm_2P2 layout to standard column major.
 */
void from_kdgemm_2P2(int ldC, const double * restrict Ck, double * restrict C)
{
    C[0 + 0*ldC] = Ck[0];
    C[1 + 1*ldC] = Ck[1];
    C[0 + 1*ldC] = Ck[2];
    C[1 + 0*ldC] = Ck[3];
}

/*
 * Convert a block from the kdgemm_4P4 layout to standard column major.
 */
void from_kdgemm_4P4(int ldC, const double* restrict Ck, double * restrict C)
{
    for (int j = 0; j < 4; j += 2) 
        for (int i = 0; i < 4; i += 2) {
            from_kdgemm_2P2(ldC, Ck, C+i+j*ldC);
            Ck += 4;
        }
}

/*
 * Convert a block from the kdgemm_8P8 layout to standard column major.
 */
void from_kdgemm_8P8(int ldC, const double* restrict Ck, double * restrict C)
{
    for (int j = 0; j < 8; j += 4) 
        for (int i = 0; i < 8; i += 4) {
            from_kdgemm_4P4(ldC, Ck, C+i+j*ldC);
            Ck += 16;
        }
}


void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C)
{
    from_kdgemm_8P8(ldC, Ck, C);
}


void kdgemm(const double * restrict A,
            const double * restrict B,
            double * restrict C)
{
    kdgemm8P8(C, A, B);
}

double* copy_optimize(const int M, const int n_blocks, double* A )
{
    int Mc = BLOCK_SIZE * n_blocks;
    double* cp = (double*) _mm_malloc(Mc * Mc * sizeof( double) , ALIGNED_SIZE);
    int i, j, I, J, ii, jj, id;
    memset(cp, 0, Mc * Mc * sizeof( double));
    for (j = 0; j < M; ++j)
    {
        J = j / BLOCK_SIZE;
        jj = j % BLOCK_SIZE;
        for (i = 0; i < M; ++i)
        {
            I = i / BLOCK_SIZE;
            ii = i % BLOCK_SIZE;
            id = (J * n_blocks + I) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;
            cp[id] = *(A++);
        }
    }
    return cp;
}

void copy_back(const int M, const int n_blocks, double* A, double* cp)
{
    int i, j, I, J, ii, jj, id;
    for (j = 0; j < M; ++j)
    {
        J = j / BLOCK_SIZE;
        jj = j % BLOCK_SIZE;
        for (i = 0; i < M; ++i)
        {
            I = i / BLOCK_SIZE;
            ii = i % BLOCK_SIZE;
            id = (J * n_blocks + I) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;
            *(cp++) = A[id];
        }
    }
}

void do_block(const double* restrict A_block, const double* restrict B_block, double* C_block)
{
    int i, j, k;
    __assume_aligned( A_block, ALIGNED_SIZE );
    __assume_aligned( B_block, ALIGNED_SIZE );
    __assume_aligned( C_block, ALIGNED_SIZE );
    for (i = 0; i < BLOCK_SIZE; ++i)
    {
        double* Ci = C_block + i * BLOCK_SIZE;
        __assume_aligned(Ci, ALIGNED_SIZE);
        for (k = 0; k < BLOCK_SIZE; ++k)
        {
            double Aik = A_block[i * BLOCK_SIZE + k];
            double* Bk = B_block + k * BLOCK_SIZE;
            __assume_aligned(Bk, ALIGNED_SIZE);
            for (j = 0; j < BLOCK_SIZE; ++j)
            {
                //#pragma vector always
                Ci[j] += Aik * Bk[j];
            }
        }
    }
}


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    double* A_cp = copy_optimize(M, n_blocks, A);
    double* B_cp = copy_optimize(M, n_blocks, B);
    double* C_cp = copy_optimize(M, n_blocks, C);
    double *A_block, *B_block, *C_block;

    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                A_block = A_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * n_blocks + bk);
                B_block = B_cp + BLOCK_SIZE * BLOCK_SIZE * (bk * n_blocks + bj);
                C_block = C_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * n_blocks + bj);
                do_block(A_block, B_block, C_block);
            }
        }
    }
    copy_back(M, n_blocks, C_cp, C);
    _mm_free(A_cp);
    _mm_free(B_cp);
    _mm_free(C_cp);
}
