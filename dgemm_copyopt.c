#include <stdlib.h>
#include <xmmintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    double* D = (double*) malloc(M * M * sizeof(double));

    // copy data from A->D to align memory accesses
    // i = column
    // j = row
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            D[j*M + i] = A[i*M + j];
        }
    }

    // j = column
    // i = row
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; j++) {
            double cij = C[j*M + i];
            for (k = 0; k < M; ++k) {
                cij += D[i*M + k] * B[j*M + k];
            }
            C[j*M + i] = cij;
        }
    }
}
