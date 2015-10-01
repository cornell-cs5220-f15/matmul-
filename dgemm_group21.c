#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

// Byte boundary to align reallocated memory 
#define ALIGNMENT ((int) 64)

// Side length threshold after which we switch to blocking.
#define BLOCK_THRESHOLD ((int) 400)

const char* dgemm_desc = "We all have our DGEMMs, but this DGEMM is mine";

inline int pad(M, pad_size) {
    return M + (pad_size - M%pad_size) * (M%pad_size > 0);
}

/*
 * Performs matrix multiplication on a single block. Note that the A matrix is 
 * stored in row major format and B is stored in column major format. 
 * The result of the multiplication is stored in C.
 *
 * Arguments:
 *  lda -- Leading dVyimension of the matrix. (Assume to be a multiple of ALIGNMENT)
 *  A   -- Pointer to block of A in row major storage and zero padded (BLOCK_SIZE * BLOCK_SIZE)
 *  B   -- Pointer to block of B in column major storage and zero padded (BLOCK_SIZE * BLOCK_SIZE)
 *  C   -- Pointer to block of C in column major storage and zero padded (BLOCK_SIZE * BLOCK_SIZE)
 */
void basic_dgemm(const int lda, const double *A, const double *B, double *C)
{
    int i, j, k;
    int jlda, ilda;
    __assume_aligned(A, ALIGNMENT);
    __assume_aligned(B, ALIGNMENT);
    __assume_aligned(C, ALIGNMENT);
    for (j = 0; j < BLOCK_SIZE; ++j) {
        jlda = j*lda;
        __assume_aligned(C, ALIGNMENT);
        for (i = 0; i < BLOCK_SIZE; ++i) {
            ilda = i*lda;
            double cij = C[jlda+i];
            __assume_aligned(A, ALIGNMENT);
            __assume_aligned(B, ALIGNMENT);
            for (k = 0; k < BLOCK_SIZE; ++k) {
                cij += A[ilda+k] * B[jlda+k];
            }
            C[jlda+i] = cij;
        }
    }
}

/*
 * Accumulates matrix multiplication on a block taken from A and B on to the appropriate block in C
 * 
 * Arguments:
 *  lda -- Leading dVyimension of the matrix. (Assume to be a multiple of ALIGNMENT)
 *  A   -- Pointer to A in row major storage and zero padded (lda * lda)
 *  B   -- Pointer to B in column major storage and zero padded (lda * lda)
 *  C   -- Pointer to C in column major storage and zero padded (lda * lda)
 */
void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    basic_dgemm(lda, A + k + i*lda, B + k + j*lda, C + i + j*lda);
}

/*
 * Transposes a matrix and stores in an freshly allocated blocked of 64 byte aligned memory
 * padded with zeros and returns the pointer to the new block.
 *
 * Arguments:
 *  A -- Pointer to the square matrix to be transposed
 *  M -- Side lenght of the matrix
 */
double* padded_transpose(const double* A, const int M) {
    int i,j;
    int pad_size = ALIGNMENT > BLOCK_SIZE ? ALIGNMENT : BLOCK_SIZE;
    const int M_padded = pad(M, pad_size);
    double* A_copy = (double*) _mm_malloc( sizeof(double) * M_padded * M_padded, ALIGNMENT );

    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[i * M_padded + j] = A[j * M + i];
        }
    }
    
    for(i=M; i<M_padded; i++) {
        for(j=M; j<M_padded; j++) {
            A_copy[i * M_padded + j] = 0;
        }
    }

    return A_copy;
}

/*
 * Copies a matrix and stores in an freshly allocated blocked of 64 byte aligned memory
 * padded with zeros and returns the pointer to the new block.
 *
 * Arguments:
 *  A -- Pointer to the square matrix to be copied
 *  M -- Side lenght of the matrix
 */
double* padded_copy(const double* A, const int M) {
    int i,j;
    int pad_size = ALIGNMENT > BLOCK_SIZE ? ALIGNMENT : BLOCK_SIZE;
    const int M_padded = pad(M, pad_size);
    double* A_copy = (double*) _mm_malloc( sizeof(double) * M_padded * M_padded, ALIGNMENT );

    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[j * M_padded + i] = A[j * M + i];
        }
    }
    
    for(i=M; i<M_padded; i++) {
        for(j=M; j<M_padded; j++) {
            A_copy[j * M_padded + i] = 0;
        }
    }

    return A_copy;
}

/*
 * Performs Double Precision General Matrix Multiplication given two dense square matrices
 * A and B stored in column major format and stores it in column major format in the allocated
 * storage area pointed to by C. 
 *
 * Arguments:
 *  M -- Side length of the square matrices
 *  A -- Pointer to an allocated block of memory with data from A in column major storage
 *  B -- Pointer to an allocated block of memory with data from B in column major storage
 *  C -- Pointer to an allocated block of memory to store C in column major storage
 */
void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
    int i,j,k;
    if( M < 1 ) {
        return;
    }

    if( M <= BLOCK_THRESHOLD ) {
        double *A_copy;

        A_copy = (double*) malloc( sizeof(double) * M * M );
        for(i = 0; i<M; i++) {
            for(j = 0; j<M; j++) {
                A_copy[i * M + j] = A[j * M + i];
            }
        }
            
        for (j = 0; j < M; ++j) {
            for (i = 0; i < M; ++i) {
                double cij = C[j*M+i];
                for (k = 0; k < M; ++k)
                    cij += A_copy[i*M+k] * B[j*M+k];
                C[j*M+i] = cij;
            }
        }

        free(A_copy);
    } else {
        int bi, bj, bk;
        double* A_copy = padded_transpose(A, M);
        double* B_copy = padded_copy(B, M);
        double* C_copy = padded_copy(C, M);

        int pad_size = ALIGNMENT > BLOCK_SIZE ? ALIGNMENT : BLOCK_SIZE;
        const int M_padded = pad(M, pad_size);
        const int n_blocks = M_padded / BLOCK_SIZE;

        for (bi = 0; bi < n_blocks; ++bi) {
            const int i = bi * BLOCK_SIZE;
            for (bj = 0; bj < n_blocks; ++bj) {
                const int j = bj * BLOCK_SIZE;
                for (bk = 0; bk < n_blocks; ++bk) {
                    const int k = bk * BLOCK_SIZE;
                    do_block(M_padded, A_copy, B_copy, C_copy, i, j, k);
                }
            }
        }
        for(i = 0; i<M; i++) {
            for(j = 0; j<M; j++) {
                C[j * M + i] = C_copy[j * M_padded + i];
            }
        }

        _mm_free(A_copy);
        _mm_free(B_copy);
        _mm_free(C_copy);
    }
}
