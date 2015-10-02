#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Copy optimized block dgemm size 64 with compiler options.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

void basic_dgemm(const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k, oi, oj, ok;
    double sum;
    for (j = 0; j < BLOCK_SIZE; ++j) {
        oj = j * BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; ++i) {
            oi = i * BLOCK_SIZE;
            
            __m256d mc = _mm256_setzero_pd();
            
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), _mm256_load_pd(B+oj   ), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), _mm256_load_pd(B+oj+ 4), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), _mm256_load_pd(B+oj+ 8), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), _mm256_load_pd(B+oj+12), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+16), _mm256_load_pd(B+oj+16), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+20), _mm256_load_pd(B+oj+20), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+24), _mm256_load_pd(B+oj+24), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+28), _mm256_load_pd(B+oj+28), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+32), _mm256_load_pd(B+oj+32), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+36), _mm256_load_pd(B+oj+36), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+40), _mm256_load_pd(B+oj+40), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+44), _mm256_load_pd(B+oj+44), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+48), _mm256_load_pd(B+oj+48), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+52), _mm256_load_pd(B+oj+52), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+56), _mm256_load_pd(B+oj+56), mc);
            mc = _mm256_fmadd_pd(_mm256_load_pd(A+oi+60), _mm256_load_pd(B+oj+60), mc);
            mc = _mm256_hadd_pd(mc, mc);
            
            // http://stackoverflow.com/questions/9775538/fastest-way-to-do-horizontal-vector-sum-with-avx-instructions
            C[oj+i] += ((double *) &mc)[0] + ((double *) &mc)[2];
        }
    }
    /*
    for (j = 0; j < M; ++j) {
        oj = j * BLOCK_SIZE;
        for (k = 0; k < M; ++k) {
            ok = k * BLOCK_SIZE;
            for (i = 0; i < M; ++i) {
                C[j*M+i] += A[k*M+i] * B[j*M+k];
            }
        }
    }
    */
    /*
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

