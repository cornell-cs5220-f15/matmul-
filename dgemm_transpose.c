#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Copy optimized transpose block dgemm size 16 with compiler options.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

void basic_dgemm(const double * restrict A, const double * restrict B, double * restrict C)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);
    
    int i, j, k, oi, oj, ok;
    double sum;
    for (i = 0; i < BLOCK_SIZE; ++i) {
        oi = i * BLOCK_SIZE;
        for (j = 0; j < BLOCK_SIZE; ++j) {
            oj = j * BLOCK_SIZE;
            sum = C[oj+i];
            for (k = 0; k < BLOCK_SIZE; ++k) {
                sum += A[oi+k] * B[oj+k];
            }
            C[oj+i] = sum;
        }
    }
    /*
    double sum;
    for (i = 0; i < BLOCK_SIZE; ++i) {
        for (j = 0; j < BLOCK_SIZE; ++j) {
            sum = 0.0;
            for (k = 0; k < BLOCK_SIZE; k += 4) {
                oi = i * BLOCK_SIZE;
                oj = j * BLOCK_SIZE;
                ok = k * BLOCK_SIZE;
                
                __m256d mma = _mm256_load_pd(A+oj+k);
                __m256d mmb = _mm256_load_pd(B+oi+k);
                
                __m256d mmr = _mm256_mul_pd(mma, mmb);
                mmr = _mm256_hadd_pd(mmr, mmr);
                
                // http://stackoverflow.com/questions/9775538/fastest-way-to-do-horizontal-vector-sum-with-avx-instructions
                sum += ((double *) &mmr)[0] + ((double *) &mmr)[2];;
            }
            C[oi+j] = sum;
        }
    }
    */
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
    int offset, offsetA, offsetB;
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            int oi = bi * BLOCK_SIZE;
            int oj = bj * BLOCK_SIZE;
            copyoffset = (bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE;
            for (j = 0; j < BLOCK_SIZE; ++j) {
                for (i = 0; i < BLOCK_SIZE; ++i) {
                    offsetA = (oi + j) + (oj + i) * M;
                    offsetB = (oi + i) + (oj + j) * M;
                    CA[copyoffset] = 0;
                    CB[copyoffset] = 0;
                    
                    // Check bounds
                    if (oi + j < M && oj + i < M) {
                        CA[copyoffset] = A[offsetA];
                    }
                    if (oi + i < M && oj + j < M) {
                        //CA[copyoffset] = A[offsetB];
                        CB[copyoffset] = B[offsetB];
                    }
                    CC[copyoffset] = 0;
                    copyoffset++;
                }
            }
        }
    }
    
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                
                // INLINE THIS FUNCTION
                basic_dgemm(
                    CA + (bi + bk * n_blocks) * BLOCK_SIZE * BLOCK_SIZE,
                    CB + (bk + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE,
                    CC + (bi + bj * n_blocks) * BLOCK_SIZE * BLOCK_SIZE
                );
            }
        }
    }
    
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

