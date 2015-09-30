#include <stdio.h>
#include <stdlib.h>

const char* dgemm_desc = "Basic, three-loop dgemm with copy optimization";

void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
    double *A_copy;
    int i,j,k;

    A_copy = (double*) malloc( sizeof(double) * M * M );
    for(i = 0; i<M; i++) {
        for(j = 0; j<M; j++) {
            A_copy[i * M + j] = A[j * M + i];
        }
    }
        
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A_copy[i*M+k] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
}

