#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Copy optimized block dgemm with compiler options and 4x4 avx.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
#endif

void basic_dgemm(const double * restrict A, const double * restrict B, double * restrict C)
{
    __assume_aligned(A, 32);
    __assume_aligned(B, 32);
    __assume_aligned(C, 32);
    
    // https://indico.cern.ch/event/327306/contribution/1/attachments/635800/875267/HaswellConundrum.pdf
    
    int i, j, k, oi, oj, ok;
    double sum;
    for (j = 0; j < BLOCK_SIZE; j+=4) {
        oj = j * BLOCK_SIZE;
        
        // Matrix B
        __m256d mb00 = _mm256_load_pd(B+oj   );
        __m256d mb04 = _mm256_load_pd(B+oj+ 4);
        __m256d mb08 = _mm256_load_pd(B+oj+ 8);
        __m256d mb12 = _mm256_load_pd(B+oj+12);
        /*
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
        */
        
        for (i = 0; i < BLOCK_SIZE; i+=4) {
            oi = i * BLOCK_SIZE;
            
            // Previous values
            __m256d mc0 = _mm256_load_pd(C+oj   );
            __m256d mc1 = _mm256_load_pd(C+oj+ 4);
            __m256d mc2 = _mm256_load_pd(C+oj+ 8);
            __m256d mc3 = _mm256_load_pd(C+oj+12);
            
            // Diagonals
            __m256d md0 = _mm256_setzero_pd();
            __m256d md1 = _mm256_setzero_pd();
            __m256d md2 = _mm256_setzero_pd();
            __m256d md3 = _mm256_setzero_pd();
            
            // Diagonal 0
            md0 = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), mb00, md0);
            md0 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), mb04, md0);
            md0 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), mb08, md0);
            md0 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), mb12, md0);
            
            // Shift matrix B (b00111001)
            mb00 = _mm256_permute4x64_pd(mb00, 0x39);
            mb04 = _mm256_permute4x64_pd(mb04, 0x39);
            mb08 = _mm256_permute4x64_pd(mb08, 0x39);
            mb12 = _mm256_permute4x64_pd(mb12, 0x39);
            
            // Diagonal 1
            md1 = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), mb00, md1);
            md1 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), mb04, md1);
            md1 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), mb08, md1);
            md1 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), mb12, md1);
            
            // Shift matrix B (b00111001)
            mb00 = _mm256_permute4x64_pd(mb00, 0x39);
            mb04 = _mm256_permute4x64_pd(mb04, 0x39);
            mb08 = _mm256_permute4x64_pd(mb08, 0x39);
            mb12 = _mm256_permute4x64_pd(mb12, 0x39);
            
            // Diagonal 2
            md2 = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), mb00, md2);
            md2 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), mb04, md2);
            md2 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), mb08, md2);
            md2 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), mb12, md2);
            
            // Shift matrix B (b00111001)
            mb00 = _mm256_permute4x64_pd(mb00, 0x39);
            mb04 = _mm256_permute4x64_pd(mb04, 0x39);
            mb08 = _mm256_permute4x64_pd(mb08, 0x39);
            mb12 = _mm256_permute4x64_pd(mb12, 0x39);
            
            // Diagonal 3
            md3 = _mm256_fmadd_pd(_mm256_load_pd(A+oi   ), mb00, md3);
            md3 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 4), mb04, md3);
            md3 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+ 8), mb08, md3);
            md3 = _mm256_fmadd_pd(_mm256_load_pd(A+oi+12), mb12, md3);
            
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
            _mm256_store_pd(C+oj   , mc0);
            _mm256_store_pd(C+oj+ 4, mc1);
            _mm256_store_pd(C+oj+ 8, mc2);
            _mm256_store_pd(C+oj+12, mc3);
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // Number of blocks total
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE? 1 : 0);
    const int n_size = n_blocks * BLOCK_SIZE;
    const int n_area = BLOCK_SIZE * BLOCK_SIZE;
    const int n_mem = n_size * n_size * sizeof(double);
    // Copied A matrix
    double * CA = (double *) malloc(n_mem);
    memset(CA, 0, n_mem);
    // Copied B matrix
    double * CB = (double *) malloc(n_mem);
    memset(CB, 0, n_mem);
    // Copied C matrix
    double * CC = (double *) malloc(n_mem);
    memset(CC, 0, n_mem);

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
                
                // INLINE THIS FUNCTION?
                basic_dgemm(
                    CA + (bi + bk * n_blocks) * n_area,
                    CB + (bk + bj * n_blocks) * n_area,
                    CC + (bi + bj * n_blocks) * n_area
                );
            }
        }
    }
    
    // Copy results back
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            int oi = bi * BLOCK_SIZE;
            int oj = bj * BLOCK_SIZE;
            copyoffset = (bi + bj * n_blocks) * n_area;
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

