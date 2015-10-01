#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Copy optimized block dgemm size 64 with compiler options.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

void basic_dgemm(const double * restrict A, const double * restrict B, double * restrict C)
{
    __assume_aligned(A, 32);
    __assume_aligned(B, 32);
    __assume_aligned(C, 32);
    
    // New C, copy New C into C
    
    int i, j, k, oi, oj, ok;
    double sum;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        oj = j * BLOCK_SIZE;
        
        // https://indico.cern.ch/event/327306/contribution/1/attachments/635800/875267/HaswellConundrum.pdf
        __m256d mb00 = _mm256_load_pd(B+oj   );
        __m256d mb04 = _mm256_load_pd(B+oj+ 4);
        __m256d mb08 = _mm256_load_pd(B+oj+ 8);
        __m256d mb12 = _mm256_load_pd(B+oj+12);
        __m256d mb16 = _mm256_load_pd(B+oj+16);
        __m256d mb20 = _mm256_load_pd(B+oj+20);
        __m256d mb24 = _mm256_load_pd(B+oj+24);
        __m256d mb28 = _mm256_load_pd(B+oj+28);
        __m256d mb32 = _mm256_load_pd(B+oj+32);
        __m256d mb36 = _mm256_load_pd(B+oj+36);
        __m256d mb40 = _mm256_load_pd(B+oj+40);
        __m256d mb44 = _mm256_load_pd(B+oj+44);
        __m256d mb48 = _mm256_load_pd(B+oj+48);
        __m256d mb52 = _mm256_load_pd(B+oj+52);
        __m256d mb56 = _mm256_load_pd(B+oj+56);
        __m256d mb60 = _mm256_load_pd(B+oj+60);
        
        for (i = 0; i < BLOCK_SIZE; ++i) {
            oi = i * BLOCK_SIZE;
            
            __m256d mc = _mm256_setzero_pd();
            
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), mb00, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), mb04, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), mb08, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), mb12, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+16), mb16, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+20), mb20, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+24), mb24, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+28), mb28, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+32), mb32, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+36), mb36, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+40), mb40, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+44), mb44, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+48), mb48, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+52), mb52, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+56), mb56, mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+60), mb60, mc);
            
            mc = _mm256_hadd_pd(mc, mc);
            
            //double * r = (double *) &mc;
            
            C[oj+i] += ((double *) &mc)[0] + ((double *) &mc)[2];
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

