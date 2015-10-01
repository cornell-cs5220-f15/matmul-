#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Copy optimized block dgemm with compiler options.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
#endif

void basic_dgemm(const double * restrict A, const double * restrict B, double * restrict C)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);
    
    // https://indico.cern.ch/event/327306/contribution/1/attachments/635800/875267/HaswellConundrum.pdf
    
    int i, j, k, t, l, oi, oj, ok;
    double sum;
    __m256d ma[BLOCK_SIZE];
    __m256d mb[BLOCK_SIZE];
    
    for (j = 0; j < BLOCK_SIZE; j+=4) {
        oj = j * BLOCK_SIZE;
        
        // Matrix B
        for (t = 0; t < BLOCK_SIZE; ++t) {
            mb[t] = _mm256_load_pd(B+oj+(4*t));
        }
        
        for (i = 0; i < BLOCK_SIZE; i+=4) {
            oi = i * BLOCK_SIZE;
            
            // Matrix A
            for (t = 0; t < BLOCK_SIZE; ++t) {
                ma[t] = _mm256_load_pd(A+oi+(4*t));
            }
            
            // Previous values
            __m256d mc0 = _mm256_load_pd(C+oj+(4*i)   );
            __m256d mc1 = _mm256_load_pd(C+oj+(4*i)+ 4);
            __m256d mc2 = _mm256_load_pd(C+oj+(4*i)+ 8);
            __m256d mc3 = _mm256_load_pd(C+oj+(4*i)+12);
            
            // Initialize diagonals
            __m256d md0 = _mm256_setzero_pd();
            __m256d md1 = _mm256_setzero_pd();
            __m256d md2 = _mm256_setzero_pd();
            __m256d md3 = _mm256_setzero_pd();
            
            // Diagonal 0
            for (t = 0; t < BLOCK_SIZE; ++t) {
                md0 = _mm256_fmadd_pd(ma[t], mb[t], md0);
            }
            
            // Shift matrix B
            /*
                shift
                0 1 2 3 -> 1 2 3 0
                    1 2 3 0 (0x39)
            */
            for (t = 0; t < BLOCK_SIZE; ++t) {
                mb[t] = _mm256_permute4x64_pd(mb[t], 0x39);
            }
            
            // Diagonal 1
            for (t = 0; t < BLOCK_SIZE; ++t) {
                md1 = _mm256_fmadd_pd(ma[t], mb[t], md1);
            }
            
            // Shift matrix B
            for (t = 0; t < BLOCK_SIZE; ++t) {
                mb[t] = _mm256_permute4x64_pd(mb[t], 0x39);
            }
            
            // Diagonal 2
            for (t = 0; t < BLOCK_SIZE; ++t) {
                md2 = _mm256_fmadd_pd(ma[t], mb[t], md2);
            }
            
            // Shift matrix B
            for (t = 0; t < BLOCK_SIZE; ++t) {
                mb[t] = _mm256_permute4x64_pd(mb[t], 0x39);
            }
            
            // Diagonal 3
            for (t = 0; t < BLOCK_SIZE; ++t) {
                md3 = _mm256_fmadd_pd(ma[t], mb[t], md3);
            }
            
            // Shift matrix B
            for (t = 0; t < BLOCK_SIZE; ++t) {
                mb[t] = _mm256_permute4x64_pd(mb[t], 0x39);
            }
            
            // Order diagonals to be stored
            /*
                0 1 2 3     0 1 2 3
                3 0 1 2 --\ 0 1 2 3
                2 3 0 1 --/ 0 1 2 3
                1 2 3 0     0 1 2 3
                
                Column 0
                    shuffle md0, md2
                    0 3 2 1    2 1 0 3
                        0 0 0 0 (0x0)
                        0 x x 0
                    
                    shuffle md1, md3
                    1 0 3 2    3 2 1 0
                        1 0 0 1 (0x9)
                        0 x x 0
                    
                    shuffle (md0, md2) (md1, md3)
                    0 x x 0    0 x x 0
                        0 0 1 1 (0xa)
                
                etc.
            */
            
            mc0 = _mm256_add_pd(
                _mm256_shuffle_pd(
                    _mm256_shuffle_pd(md0, md2, 0x0),
                    _mm256_shuffle_pd(md3, md1, 0x9),
                    0xc
                ), mc0);
            mc1 = _mm256_add_pd(
                _mm256_shuffle_pd(
                    _mm256_shuffle_pd(md1, md3, 0x0),
                    _mm256_shuffle_pd(md0, md2, 0x9),
                    0xc
                ), mc1);
            mc2 = _mm256_add_pd(
                _mm256_shuffle_pd(
                    _mm256_shuffle_pd(md2, md0, 0x0),
                    _mm256_shuffle_pd(md1, md3, 0x9),
                    0xc
                ), mc2);
            mc3 = _mm256_add_pd(
                _mm256_shuffle_pd(
                    _mm256_shuffle_pd(md3, md1, 0x0),
                    _mm256_shuffle_pd(md2, md0, 0x9),
                    0xc
                ), mc3);
            
            // Store back into result
            _mm256_store_pd(C+oj+(4*i)   , mc0);
            _mm256_store_pd(C+oj+(4*i)+ 4, mc1);
            _mm256_store_pd(C+oj+(4*i)+ 8, mc2);
            _mm256_store_pd(C+oj+(4*i)+12, mc3);
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
            for (j = 0; j < BLOCK_SIZE; j+=4) {
                for (i = 0; i < BLOCK_SIZE; i+=4) {
                    offsetA = (oi + j) + (oj + i) * M;
                    offsetB = (oi + i) + (oj + j) * M;
                    CA[copyoffset] = 0;
                    CB[copyoffset] = 0;
                    
                    // Check bounds
                    if (oi + j < M && oj + i < M) {
                        //CA[copyoffset] = A[offsetA];
                        CB[copyoffset] = B[offsetA];
                    }
                    if (oi + i < M && oj + j < M) {
                        CA[copyoffset] = A[offsetB];
                        //CB[copyoffset] = B[offsetB];
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

