#include <stdlib.h>

const char* dgemm_desc = "Copy optimized block dgemm size 16.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

void basic_dgemm(const double *A, const double *B, double *C)
{
    int i, j, k, oi, oj, ok;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        oj = j * BLOCK_SIZE;
        for (k = 0; k < BLOCK_SIZE; ++k) {
            ok = k * BLOCK_SIZE;
            for (i = 0; i < BLOCK_SIZE; ++i) {
                C[oj+i] += A[ok+i] * B[oj+k];
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Number of blocks total
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE? 1 : 0);
    const int n_size = n_blocks * BLOCK_SIZE;
    const int n_mem = n_size * n_size * sizeof(double);
    // Copied A matrix
    double * CA = (double *) malloc(n_mem);
    // Copied B matrix
    double * CB = (double *) malloc(n_mem);
    // Copied C matrix
    double * CC = (double *) malloc(n_mem);

    // Initialize matrices
    int bi, bj, bk, i, j, k;

    int copyoffset;
    int offset;
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            int oi = bi * BLOCK_SIZE;
            int oj = bj * BLOCK_SIZE;
            copyoffset = (bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE;
            for (j = 0; j < BLOCK_SIZE; ++j) {
                for (i = 0; i < BLOCK_SIZE; ++i) {
                    offset = (oi + i) + (oj + j) * M;
                    // Check bounds
                    if (oi + i < M && oj + j < M) {
                        CA[copyoffset] = A[offset];
                        CB[copyoffset] = B[offset];
                        CC[copyoffset] = 0;
                        offset++;
                    }
                    else {
                        CA[copyoffset] = 0;
                        CB[copyoffset] = 0;
                        CC[copyoffset] = 0;
                    }
                    copyoffset++;
                }
            }
        }
    }
    
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                //CC[(bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE]++;
                //*
                basic_dgemm(
                    CA + (bi + bk * n_blocks) * BLOCK_SIZE * BLOCK_SIZE,
                    CB + (bk + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE,
                    CC + (bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE
                );
                //*/
            }
        }
    }
    
    /*
    for (bj = 0; bj < n_size; bj++) {
        for (bi = 0; bi < n_size; bi++) {
            printf("%.1f ", CC[bi + bj * n_size]);
        }
        printf("\n");
    }
    */

    // Copy results back
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            int oi = bi * BLOCK_SIZE;
            int oj = bj * BLOCK_SIZE;
            copyoffset = (bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE;
            for (j = 0; j < BLOCK_SIZE; ++j) {
                for (i = 0; i < BLOCK_SIZE; ++i) {
                    offset = (oi + i) + (oj + j) * M;
                    // Check bounds
                    if (oi + i < M && oj + j < M) {
                        C[offset] = CC[copyoffset];
                    }
                    copyoffset++;
                }
            }
        }
    }
}

