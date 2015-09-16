#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

    double A [16] = { 4, 3, 3, 3, 4, 2, 3, 3, 1, 2, 4, 3, 2, 2, 2, 4 };
    double B [16] = { 3, 1, 4, 2, 3, 2, 2, 1, 4, 4, 3, 1, 2, 1, 2, 3 };
    double* C = (double*) calloc (16,  sizeof(double) );
    double* C_c = (double*) calloc (4, sizeof(double) );
    double* A_c = (double*) calloc (4, sizeof(double) );
    int i,j, k;
    int lda_copy = 4;

    for(i = 0; i<lda_copy; i++) {

        double b_copy;
        __m256d SSEa, SSEb, SSEc, Result, Int;
        SSEc = _mm256_load_pd(C + i*lda_copy);

        SSEa = _mm256_load_pd(A);
        b_copy = B[i * lda_copy];
        SSEb = _mm256_broadcast_sd(&b_copy);
        SSEc = _mm256_add_pd(_mm256_mul_pd(SSEa, SSEb), SSEc);

        SSEa = _mm256_load_pd(A + 1*lda_copy);
        b_copy = B[i * lda_copy + 1];
        SSEb = _mm256_broadcast_sd(&b_copy);
        SSEc = _mm256_add_pd(_mm256_mul_pd(SSEa, SSEb), SSEc);

        SSEa = _mm256_load_pd(A + 2*lda_copy);
        b_copy = B[i * lda_copy + 2];
        SSEb = _mm256_broadcast_sd(&b_copy);
        SSEc = _mm256_add_pd(_mm256_mul_pd(SSEa, SSEb), SSEc);

        SSEa = _mm256_load_pd(A + 3*lda_copy);
        b_copy = B[i * lda_copy + 3];
        SSEb = _mm256_broadcast_sd(&b_copy);
        SSEc = _mm256_add_pd(_mm256_mul_pd(SSEa, SSEb), SSEc);

        _mm256_store_pd(C + i * lda_copy, SSEc);
    }

    for(i = 0; i<16; i++) {
        printf("%05.2f ", C[i]);
        if((i+1)%4 == 0) printf("\n");
    }

}
