const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>

// number of doubles in one row of a block.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// number of blocks that fit in our L2 cache.
#ifndef NUM_BLOCKS_PER_L2
#define NUM_BLOCKS_PER_L2 2
#endif

// number of doubles that fit into our L2 cache.
#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 64
#endif

#ifndef USE_L2_BLOCKING
#define USE_L2_BLOCKING 1
#endif
 
/* Copies the input matrix so that we optimize for cache hits.
*  We assume that the input matrix is arranged in column-major order.
*  arguments:
*       const int num_blocks : number of blocks in one dimension
*       const int M          : length of one column in M. 
*       const double *A      : the matrix in column major order. We assume that it is square.
*
*  output:
*       double *M : a copy of the matrix *A, but blocks are contiguous in memory and each
*                   block is arranged in row-major order. Then blocks are arranged in row major order.
*/
double* copy_optimize_rowmajor( const int num_blocks, const int M, const double *A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    // double* out = ( double* ) _mm_malloc( out_dim * out_dim * sizeof( double ) , 64);
    double* out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            if( i < M && j < M )
            {
                in_idx  = j * M + i;
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = 0.0;
            }
        }
    }
    return out;
}


double* copy_optimize_colmajor( const int num_blocks, const int M, const double *A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    // double* out = ( double* ) _mm_malloc( out_dim * out_dim * sizeof( double ) , 64);
    double* out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = ( J * num_blocks + I ) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            if( i < M && j < M )
            {
                in_idx  = j * M + i;
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = 0.0;
            }
        }
    }

    return out;
}

double* copy_optimize_colmajor_L2( const int num_blocks_L2, const int M, const double *A )
{
    int out_dim = num_blocks_L2 * BLOCK_SIZE_L2;
    double *out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );

    int k, l; // specific column, row of the input matrix.
    int kk, ll; // {k, l} % BLOCK_SIZE_L2
    int I, J; // specific column, row of the L2 block. < num_blocks_L2
    int i, j; // specific column, row of the block inside of the L2 block. < NUM_BLOCKS_PER_L2
    int ii, jj; // specific column, row of the double inside of the block. < BLOCK_SIZE
    int out_idx, in_idx;
    for( k = 0; k < out_dim; ++k )
    {
        // figure out what L2 block column we're in.
        I = k / BLOCK_SIZE_L2;

        // figure out what L1 block column within the L2 block we're in.
        kk = k % BLOCK_SIZE_L2;
        i = kk / BLOCK_SIZE;

        // figure out what column within the L1 block we're in
        ii = kk % BLOCK_SIZE;

        in_idx = k * M;
        for( l = 0; l < out_dim; ++l )
        {
            // indexing for columns
            J = l / BLOCK_SIZE_L2;
            ll = l % BLOCK_SIZE_L2;
            j = ll / BLOCK_SIZE;
            jj = ll % BLOCK_SIZE;

            // obtain out_index
            out_idx = ( I * num_blocks_L2 + J ) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2 + 
                      ( i * NUM_BLOCKS_PER_L2 + j ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;

            if( k < M && l < M )
            {
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = 0.0;
            }
            ++in_idx;
        }
    }

    return out;
}

double* copy_optimize_rowmajor_L2( const int num_blocks_L2, const int M, const double *A )
{
    int out_dim = num_blocks_L2 * BLOCK_SIZE_L2;
    double *out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );

    int k, l; // specific column, row of the input matrix.
    int kk, ll; // {k, l} % BLOCK_SIZE_L2
    int I, J; // specific column, row of the L2 block. < num_blocks_L2
    int i, j; // specific column, row of the block inside of the L2 block. < NUM_BLOCKS_PER_L2
    int ii, jj; // specific column, row of the double inside of the block. < BLOCK_SIZE
    int out_idx, in_idx;
    for( k = 0; k < out_dim; ++k )
    {
        // figure out what L2 block column we're in.
        I = k / BLOCK_SIZE_L2;

        // figure out what L1 block column within the L2 block we're in.
        kk = k % BLOCK_SIZE_L2;
        i = kk / BLOCK_SIZE;

        // figure out what column within the L1 block we're in
        ii = kk % BLOCK_SIZE;

        in_idx = k * M;
        for( l = 0; l < out_dim; ++l )
        {
            // indexing for columns
            J = l / BLOCK_SIZE_L2;
            ll = l % BLOCK_SIZE_L2;
            j = ll / BLOCK_SIZE;
            jj = ll % BLOCK_SIZE;

            // obtain out_index
            out_idx = ( J * num_blocks_L2 + I ) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2 + 
                      ( j * NUM_BLOCKS_PER_L2 + i ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;

            if( k < M && l < M )
            {
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = 0.0;
            }
            ++in_idx;
        }
    }

    return out;
}


/* Performs a block multiply on block I, J in matrix A and with block K, L in matrix B.
*  A is block row major and B is block column major.
*  C is outputted in block column major order.
*  C = C + AB
*/
void block_multiply_kernel( const int M, double *A_block, double *B_block, double *C_block )
{
    int ai, aj, bj;
    for( ai = 0; ai < BLOCK_SIZE; ++ai )
    {
        double* C_row = C_block + ai * BLOCK_SIZE;
        // __assume_aligned( C_row, 64 );
        for( aj = 0; aj < BLOCK_SIZE; ++aj )
        {
            double A_element = A_block[ai * BLOCK_SIZE + aj];
            double* B_row = B_block + aj * BLOCK_SIZE;
            // __assume_aligned( B_row, 64 );
            for( bj = 0; bj < BLOCK_SIZE; ++bj )
            {
                #pragma vector always
                C_row[bj] += B_row[bj] * A_element;
            }
        }
    }
}



/* Copies matrix A so that it is back to normal column major order. 
*  A is block column major on input.
*/
void copy_normal_rowmajor( const int num_blocks, const int M, const double *A, double* out )
{
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int in_idx;
    int out_idx;
    for( i = 0; i < M; ++i )
    {
        int I = i / BLOCK_SIZE;
        int ii = i % BLOCK_SIZE;
        for( j = 0; j < M; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = j * M + i;
            in_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            out[ out_idx ] = A[ in_idx ];
        }
    }
}

double* copy_normal_rowmajor_L2( const int num_blocks_L2, const int M, const double *A, double *out )
{
    int k, l; // specific column, row of the input matrix.
    int kk, ll; // {k, l} % BLOCK_SIZE_L2
    int I, J; // specific column, row of the L2 block. < num_blocks_L2
    int i, j; // specific column, row of the block inside of the L2 block. < NUM_BLOCKS_PER_L2
    int ii, jj; // specific column, row of the double inside of the block. < BLOCK_SIZE
    int out_idx, in_idx;
    for( k = 0; k < M; ++k )
    {
        // figure out what L2 block column we're in.
        I = k / BLOCK_SIZE_L2;

        // figure out what L1 block column within the L2 block we're in.
        kk = k % BLOCK_SIZE_L2;
        i = kk / BLOCK_SIZE;

        // figure out what column within the L1 block we're in
        ii = kk % BLOCK_SIZE;

        out_idx = k * M;
        for( l = 0; l < M; ++l )
        {
            // indexing for columns
            J = l / BLOCK_SIZE_L2;
            ll = l % BLOCK_SIZE_L2;
            j = ll / BLOCK_SIZE;
            jj = ll % BLOCK_SIZE;

            // obtain out_index
            in_idx = ( J * num_blocks_L2 + I ) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2 + 
                      ( j * NUM_BLOCKS_PER_L2 + i ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;

            out[ out_idx ] = A[ in_idx ];
            ++out_idx;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    if( USE_L2_BLOCKING )
    {
        const int num_blocks_L2 = M / BLOCK_SIZE_L2 + (int) ( M % BLOCK_SIZE_L2 != 0 );
        double* A_copied = copy_optimize_rowmajor_L2( num_blocks_L2, M, A );
        double* B_copied = copy_optimize_colmajor_L2( num_blocks_L2, M, B );
        double* C_copied = copy_optimize_rowmajor_L2( num_blocks_L2, M, C );

        int I, J, K; // indices for L2 blocks
        int i, j, k; // indices for L1 blocks
        double *A_block2, *B_block2, *C_block2;
        double *A_block, *B_block, *C_block;
        for( I = 0; I < num_blocks_L2; ++I )
        {
            for( J = 0; J < num_blocks_L2; ++J )
            {
                C_block2 = C_copied + ( I * num_blocks_L2 + J) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2;
                for( K = 0; K < num_blocks_L2; ++K )
                {
                    A_block2 = A_copied + ( I * num_blocks_L2 + K ) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2;
                    B_block2 = B_copied + ( J * num_blocks_L2 + K ) * BLOCK_SIZE_L2 * BLOCK_SIZE_L2;
                    for( i = 0; i < NUM_BLOCKS_PER_L2; ++i )
                    {
                        for( j = 0; j < NUM_BLOCKS_PER_L2; ++j )
                        {
                            C_block = C_block2 + ( i * NUM_BLOCKS_PER_L2 + j ) * BLOCK_SIZE * BLOCK_SIZE;
                            // __assume_aligned( C_block, 64 );
                            for( k = 0; k < NUM_BLOCKS_PER_L2; ++k )
                            {
                                A_block = A_block2 + ( i * NUM_BLOCKS_PER_L2 + k ) * BLOCK_SIZE * BLOCK_SIZE;
                                B_block = B_block2 + ( j * NUM_BLOCKS_PER_L2 + k ) * BLOCK_SIZE * BLOCK_SIZE;
                                // __assume_aligned( A_block, 64 );
                                // __assume_aligned( B_block, 64 );
                                block_multiply_kernel( M, A_block, B_block, C_block );
                            }
                        }
                    }
                }
            }
        }

        copy_normal_rowmajor_L2( num_blocks_L2, M, C_copied, C );
        
        // _mm_free(A_copied);
        free(A_copied);
        // _mm_free(B_copied);
        free(B_copied);
        // _mm_free(C_copied);
        free(C_copied);
    }
    else
    {
        const int num_blocks = M / BLOCK_SIZE + (int) ( M % BLOCK_SIZE != 0 );
        double* A_copied = copy_optimize_rowmajor( num_blocks, M, A );
        double* B_copied = copy_optimize_colmajor( num_blocks, M, B );
        double* C_copied = copy_optimize_rowmajor( num_blocks, M, C );

        int I, J, K;
        int A_idx, B_idx, C_idx;
        double *A_block, *B_block, *C_block;
        for( I = 0; I < num_blocks; ++I ) 
        {
            for( J = 0; J < num_blocks; ++J )
            {
                // index of first element in matrix C.
                C_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE;
                C_block = C_copied + C_idx;
                // __assume_aligned( C_block, 64 );
                
                for( K = 0; K < num_blocks; ++K )
                {
                    // index of first element to be multiplied in matrix A
                    A_idx = ( I * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

                    // index of first element to be multiplied in matrix B
                    B_idx = ( J * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

                    A_block = A_copied + A_idx;
                    B_block = B_copied + B_idx;
                    // __assume_aligned( A_block, 64 );
                    // __assume_aligned( B_block, 64 );
                    block_multiply_kernel( M, A_block, B_block, C_block );
                }
            }
        }

        copy_normal_rowmajor( num_blocks, M, C_copied, C );
        
        // _mm_free(A_copied);
        free(A_copied);
        // _mm_free(B_copied);
        free(B_copied);
        // _mm_free(C_copied);
        free(C_copied);
    }
}

