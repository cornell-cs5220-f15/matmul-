#include <stdlib.h>
#include <xmmintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

union U {
    __m128d v;    // SSE 4 x float vector
    double a[2];  // scalar array of 4 floats
};

double vectorGetByIndex(__m128d V, unsigned int i)
{
    union U u;

    u.v = V;
    return u.a[i];
}

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

    // i = column
    // j = row
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; j++) {
            double cij = C[i*M + j];
            for (k = 0; k < M; ++k) {
                cij += D[j*M + k] * B[i*M + k];
            }
            C[i*M + j] = cij;
        }
    }
}
