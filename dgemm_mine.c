#include <stdlib.h>

const char* dgemm_desc = "Basic, three-loop dgemm.";

void make_transpose(const int M, const double *A, double *out)
{
    int i, j;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            out[j*M + i] = A[i*M + j];
        }
    }
}


void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
    int i, j, k;
    double *A_T = (double*)malloc(M * M * sizeof(double));
    make_transpose(M, A, A_T);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A_T[i*M+k] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
    free(A_T);
}
