#include <unistd.h> // sysconf
#include <math.h>   // sqrt

const char* dgemm_desc = "Simple blocked dgemm.";

/*
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif
*/

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
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k){
            double bkj = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                C[j*lda+i] += A[k*lda+i] * bkj;
            }
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k,
              const int BLOCK_SIZE)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}



void basic_transpose(const int lda, const int M, const int N, double* mat) {
    int i, j;
    double tmp;
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            tmp = mat[j*lda+i];
            mat[j*lda+i] = mat[i*lda+j];
            mat[i*lda+j] = tmp;
        }
    }
}

void transpose_block(const int lda,
                     const int i, const int j,
                     const int BLOCK_SIZE,
                     double* mat) {
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    basic_transpose(lda, M, N, mat + i + j*lda);
}

void transpose(const int lda, const int n_blocks, const int BLOCK_SIZE, double* mat) {
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            transpose_block(lda, i, j, BLOCK_SIZE, mat);
        }
    }
}

#include <stdio.h>

inline void naive_transpose(int M, double* mat) {
    int c, r;
    double tmp;
    for(int c = 0; c < M; ++c) {
        for(int r = 0; r < M; ++r) {
            tmp = mat[c*M+r];
            mat[c*M+r] = mat[r*M+c];
            mat[r*M+c] = tmp;
        }
    }
}


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
    size_t num_pages = (size_t) sysconf(_SC_PHYS_PAGES);
    size_t blocked_bytes = (page_size * num_pages) / ((size_t) 24);

    long double floating_blocked_bytes = (long double) blocked_bytes;
    long double block_size = sqrt(floating_blocked_bytes);

    size_t floor_block_size = (size_t) block_size;
    const int BLOCK_SIZE = 1024; //(int) floor_block_size;

    /*
    printf("PAGE_SIZE:   %zd\n", page_size);
    printf("NUM_PAGES:   %zd\n", num_pages);
    printf("BLOCK_BYTES: %zd\n", blocked_bytes);
    printf("SQRT_BYTES:  %LF\n", block_size);
    printf("BLOCK_SIZE:  %d\n\n", BLOCK_SIZE);
    */
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);

    double* bad_idea = (double *)(&(*A));
    // transpose(M, n_blocks, BLOCK_SIZE, bad_idea);
    naive_transpose(M, bad_idea);

    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k, BLOCK_SIZE);
            }
        }
    }

    naive_transpose(M, bad_idea);
    //transpose(M, n_blocks, BLOCK_SIZE, bad_idea);
}

