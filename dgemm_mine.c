const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 2
#endif

#ifndef BLOCK_SIZE_SQ
#define BLOCK_SIZE_SQ 4
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
    double* out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );
    int i, j; // specific column, row for the matrix. < M
    int I, J; // specific column, row for the block < num_blocks
    int k, l; // specific column, row within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        k = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            l = j % BLOCK_SIZE;
            out_idx = ( J * num_blocks + I ) * BLOCK_SIZE_SQ + l * BLOCK_SIZE + k;
            if( i < M && j < M )
            {
                in_idx  = j + i * M;
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

/* Copies the input matrix so that we optimize for cache hits.
*  We assume that the input matrix is arranged in column-major order.
*  arguments:
*       const int num_blocks : number of blocks in one dimension
*       const int M          : length of one column in M. 
*       const double *A      : the matrix in column major order. We assume that it is square.
*
*  output:
*       double *M : a copy of the matrix *A, but blocks are contiguous in memory and each
*                   block is arranged in column-major order. Then blocks are arranged in column major order.
*/
double* copy_optimize_colmajor( const int num_blocks, const int M, const double *A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    double* out = ( double* ) malloc( out_dim * out_dim * sizeof( double ) );
    int i, j; // specific column, row for the matrix. < M
    int I, J; // specific column, row for the block < num_blocks
    int k, l; // specific column, row within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        k = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            l = j % BLOCK_SIZE;
            out_idx = ( I * num_blocks + J ) * BLOCK_SIZE_SQ + k * BLOCK_SIZE + l;
            if( i < M && j < M )
            {
                in_idx  = j + i * M;
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
void block_multiply_kernel( const int num_blocks, const int M, const int I, const int J,
                            const int K, const int L, const double *A, const double *B, double *C )
{
    // index of first element to be multiplied in matrix A
    const int A_idx = ( J * num_blocks + I ) * BLOCK_SIZE_SQ;

    // index of first element to be multiplied in matrix B
    const int B_idx = ( K * num_blocks + L ) * BLOCK_SIZE_SQ;

    // index of first element in matrix C.
    const int C_idx = ( K * num_blocks + J ) * BLOCK_SIZE_SQ;

    int i, j;
    int i_BLOCK_SIZE = 0;
    for( i = 0; i < BLOCK_SIZE; ++i )
    {
        for( j = 0; j < BLOCK_SIZE_SQ; ++j )
        {
            int b_offset = j % BLOCK_SIZE + i_BLOCK_SIZE;
            int c_offset = j / BLOCK_SIZE + i_BLOCK_SIZE;
            C[ c_offset + C_idx ] += A[ A_idx + j ] * B[ B_idx + b_offset ];
        }
        i_BLOCK_SIZE += BLOCK_SIZE;
    }
}

/* Performs a block add operation.
*  Adds together block I, J in matrices A and B and into C.
*  A is block row major, B is block column major, C is block column major.
*/
void block_add_kernel( const int num_blocks, const int M, const int I, const int J,
                       const double *A, const double *B, double *C )
{
    const int A_idx = ( J * num_blocks + I ) * BLOCK_SIZE_SQ;
    const int B_idx = ( I * num_blocks + J ) * BLOCK_SIZE_SQ;
    const int C_idx = B_idx;

    int i;
    for( i = 0; i < BLOCK_SIZE_SQ; ++i )
    {
        int a_offset = ( i % BLOCK_SIZE ) * BLOCK_SIZE + ( i / BLOCK_SIZE );
        C[ C_idx + i ] = B[ B_idx + i ] + A[ A_idx + a_offset ];
    }
}

/* Copies matrix A so that it is back to normal column major order. 
*  A is block column major on input.
*/
void copy_normal_colmajor( const int num_blocks, const int M, const double *A, double* out )
{
    const int extras = num_blocks * BLOCK_SIZE - M;

    int i, j; // specific column, row for the matrix. < M
    int I, J; // specific column, row for the block < num_blocks
    int k, l; // specific column, row within the block. < BLOCK_SIZE
    int in_idx;
    int out_idx;
    for( i = 0; i < M; ++i )
    {
        int I = i / BLOCK_SIZE;
        int k = i % BLOCK_SIZE;
        for( j = 0; j < M; ++j )
        {
            J = j / BLOCK_SIZE;
            l = j % BLOCK_SIZE;
            in_idx = ( I * num_blocks + J ) * BLOCK_SIZE_SQ + k * BLOCK_SIZE + l;
            out_idx  = j + i * M;
            out[ out_idx ] = A[ in_idx ];
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int num_blocks = M / BLOCK_SIZE + (int) ( M % BLOCK_SIZE != 0 );
    const double* A_copied = copy_optimize_rowmajor( num_blocks, M, A );
    const double* B_copied = copy_optimize_colmajor( num_blocks, M, B );
    double* C_copied = copy_optimize_colmajor( num_blocks, M, C );

    // printf( "A = \n" );
    // for( int l = 0; l < M * M; ++ l )
    // {
    //     printf( " %f, ", A[l] );
    // }

    // printf( "\n\n" );

    // printf( "A_copied = \n" );
    // for( int l = 0; l < M * M; ++ l )
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
    // for( int l = 0; l < M * M; ++ l )
    // {
    //     printf( " %f, ", B_copied[l] );
    // }

    // printf( "\n\n" );

    int i, j, k;
    for( i = 0; i < num_blocks; ++i ) 
    {
        for( j = 0; j < num_blocks; ++j )
        {
            for( k = 0; k < num_blocks; ++k )
            {
                block_multiply_kernel( num_blocks, M, k, i,
                                       j, k, A_copied, B_copied, C_copied );
            }
        }
    }

    copy_normal_colmajor( num_blocks, M, C_copied, C );

}
