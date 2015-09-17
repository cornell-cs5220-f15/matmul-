const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32 
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
    double* out = ( double* ) _mm_malloc( out_dim * out_dim * sizeof( double ) , 64);
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
    double* out = ( double* ) _mm_malloc( out_dim * out_dim * sizeof( double ) , 64);
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


/* Performs a block multiply on block I, J in matrix A and with block K, L in matrix B.
*  A is block row major and B is block column major.
*  C is outputted in block column major order.
*  C = C + AB
*/
void block_multiply_kernel( const int num_blocks, const int M, const int I, const int J, const int K, const double *A, const double *B, double *C )
{
    // index of first element to be multiplied in matrix A
    const int A_idx = ( I * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

    // index of first element to be multiplied in matrix B
    const int B_idx = ( J * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

    // index of first element in matrix C.
    const int C_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE;

    const double* A_block = A + A_idx;
    const double* B_block = B + B_idx;
    const double* C_block = C + C_idx;
    __assume_aligned( A_block, 64 );
    __assume_aligned( B_block, 64 );
    __assume_aligned( C_block, 64 );

    int ai, aj, bj;
    for( ai = 0; ai < BLOCK_SIZE; ++ai )
    {
        double* C_row = C_block + ai * BLOCK_SIZE;
        __assume_aligned( C_row, 64 );
        for( aj = 0; aj < BLOCK_SIZE; ++aj )
        {
            double A_element = A_block[ai * BLOCK_SIZE + aj];
            double* B_row = B_block + aj * BLOCK_SIZE;
            __assume_aligned( B_row, 64 );
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

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int num_blocks = M / BLOCK_SIZE + (int) ( M % BLOCK_SIZE != 0 );
    double* A_copied = copy_optimize_rowmajor( num_blocks, M, A );
    double* B_copied = copy_optimize_colmajor( num_blocks, M, B );
    double* C_copied = copy_optimize_rowmajor( num_blocks, M, C );

    // printf( "A = \n" );
    // for( int l = 0; l < M * M; ++ l )
    // {
    //     printf( " %f, ", A[l] );
    // }
    // printf( "\n\n" );

    // printf( "A_copied = \n" );
    // for( int l = 0; l < num_blocks * num_blocks * BLOCK_SIZE * BLOCK_SIZE; ++ l )
    // {
    //     printf( " %f, ", A_copied[l] );
    // }
    // printf( "\n\n" );

    // printf( "B = \n" );
    // for( int l = 0; l < M * M; ++ l )
    // {
    //     printf( " %f, ", B[l] );
    // }
    // printf( "\n\n" );

    // printf( "B_copied = \n" );
    // for( int l = 0; l < num_blocks * num_blocks * BLOCK_SIZE * BLOCK_SIZE; ++ l )
    // {
    //     printf( " %f, ", B_copied[l] );
    // }
    // printf( "\n\n" );

    int I, J, K;
    for( I = 0; I < num_blocks; ++I ) 
    {
        for( J = 0; J < num_blocks; ++J )
        {
            for( K = 0; K < num_blocks; ++K )
            {
                block_multiply_kernel( num_blocks, M, I, J, K, A_copied, B_copied, C_copied );
            }
        }
    }

    // printf( "C_copied = \n" );
    // for( int l = 0; l < num_blocks * num_blocks * BLOCK_SIZE * BLOCK_SIZE; ++ l )
    // {
    //     printf( " %f, ", C_copied[l] );
    // }
    // printf( "\n\n" );

    copy_normal_rowmajor( num_blocks, M, C_copied, C );

    // printf( "C = \n" );
    // for( int l = 0; l < M * M; ++ l )
    // {
    //     printf( " %f, ", C[l] );
    // }
    // printf( "\n\n" );
    
    _mm_free(A_copied);
    _mm_free(B_copied);
    _mm_free(C_copied);
}

