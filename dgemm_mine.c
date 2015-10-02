const char* dgemm_desc = "My Awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 96)
#endif

/*
 A is M-by-K
 B is K-by-N
 C is M-by-N
 lda is the leading dimension of the matrix (the M of square_dgemm).
 */

//Copy Optimization: Memory Aligned Buffers for Matrix Blocks
double A_buf[ BLOCK_SIZE * BLOCK_SIZE ] __attribute__((aligned( 32 )));
double B_buf[ BLOCK_SIZE * BLOCK_SIZE ] __attribute__((aligned( 32 )));
double C_buf[ BLOCK_SIZE * BLOCK_SIZE ] __attribute__((aligned( 32 )));

//Restrict Pointers: restrict is used to limit effects of pointer aliasing, aiding optimizations
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    
    //---Copy Blocks of A,B and C into A_buf, B_buf and C_buf respectively---//
    
    //Copy Block of A into A_buf in row-major form to aid vectorization of matrix multiply's innermost loop
    for( k = 0; k < K; ++k ) {
        for( i = 0; i < M; ++i ) {
            A_buf[ K * i + k ] = A[ i + lda * k ];
        }
    }
    
    for( j = 0; j < N; ++j ) {
        for( k = 0; k < K; ++k ) {
            B_buf[ k + K * j ] = B[ k + j * lda ];
        }
    }
    
    for( j = 0; j < N; ++j ) {
        for( i = 0; i < M; ++i ) {
            C_buf[ i + M * j ] = C[ i + lda * j ];
        }
    }
    
    //---End of Copy Optimization---//
    
    //---Vectorized Matrix Multiply Kernel: Dot Product of Rows of A_buf with Columns of B_buf
    // to produce Columns of C_buf---//
    
    for ( i = 0; i < M; ++i ) {
        for ( j = 0; j < N; ++j ) {
            
            #pragma vector aligned
            for ( k = 0; k < K; ++k ) {
                C_buf[ i + M * j ] += A_buf[ K * i + k ] * B_buf[ k + K * j ];
            }
            
        }
    }
    
    //---End of Matrix Multiply Kernel---//
    
    //---Copy back the computed C_buf block into C---//
    
    for( j = 0; j < N; ++j ) {
        for( i = 0; i < M; ++i ) {
            C[ i + lda * j ] = C_buf[ i + M * j ];
        }
    }
    
    //---End of Copy Back---//
}

void do_block(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k)
{
    const int M = ( i + BLOCK_SIZE > lda ? lda-i : BLOCK_SIZE );
    const int N = ( j + BLOCK_SIZE > lda ? lda-j : BLOCK_SIZE );
    const int K = ( k + BLOCK_SIZE > lda ? lda-k : BLOCK_SIZE );
    
    basic_dgemm( lda, M, N, K,
                A + i + lda * k, B + k + lda * j, C + i + lda * j );
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + ( M % BLOCK_SIZE ? 1 : 0 );
    int bi, bj, bk;
    for ( bi = 0; bi < n_blocks; ++bi ) {
        const int i = bi * BLOCK_SIZE;
        for ( bj = 0; bj < n_blocks; ++bj ) {
            const int j = bj * BLOCK_SIZE;
            for ( bk = 0; bk < n_blocks; ++bk ) {
                const int k = bk * BLOCK_SIZE;
                do_block( M, A, B, C, i, j, k );
            }
        }
    }
}
