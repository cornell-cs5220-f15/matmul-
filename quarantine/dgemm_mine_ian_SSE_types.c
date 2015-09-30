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
        for (j = 0; j < M-1; j+=2) {
            __m128d cij = _mm_set_pd(C[i*M+j], C[i*M+j+1]);
            __m128d temp;
            for (k = 0; k < M; ++k) {
                __m128d Bvect = _mm_set_pd(B[i*M + k], B[i*M + k]);
                __m128d Avect = _mm_set_pd(D[j*M + k], D[(j+1)*M + k]);
                temp = _mm_mul_pd(Avect, Bvect);
                cij = _mm_add_pd(cij, temp);
            }
            C[i*M + j] = vectorGetByIndex(cij, 1);
            C[i*M + j + 1] = vectorGetByIndex(cij, 0);
        }
        if (j == M-1) {
            double cij = C[i*M + j];
            for (k = 0; k < M; ++k) {
                cij += D[j*M + k] * B[i*M + k];
            }
            C[i*M + j] = cij;
        }
    }
}
