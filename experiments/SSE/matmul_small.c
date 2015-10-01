#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

    double A [16] = { 4, 3, 3, 3, 4, 2, 3, 3, 1, 2, 4, 3, 2, 2, 2, 4 };
    double B [16] = { 3, 1, 4, 2, 3, 2, 2, 1, 4, 4, 3, 1, 2, 1, 2, 3 };
    double* C = (double*) calloc (16,  sizeof(double) );
    int i,j, k;

    double b_copy;
    __m256d ymm_a, ymm_b, ymm_c;
    for(i = 0; i<4; i++) {
        ymm_c = _mm256_load_pd(C + i*4);

        ymm_a = _mm256_load_pd(A);
        b_copy = B[i * 4];
        ymm_b = _mm256_broadcast_sd(&b_copy);
        ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

        ymm_a = _mm256_load_pd(A + 1*4);
        b_copy = B[i * 4 + 1];
        ymm_b = _mm256_broadcast_sd(&b_copy);
        ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

        ymm_a = _mm256_load_pd(A + 2*4);
        b_copy = B[i * 4 + 2];
        ymm_b = _mm256_broadcast_sd(&b_copy);
        ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

        ymm_a = _mm256_load_pd(A + 3*4);
        b_copy = B[i * 4 + 3];
        ymm_b = _mm256_broadcast_sd(&b_copy);
        ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);

        _mm256_store_pd(C + i * 4, ymm_c);
    }

    for(i = 0; i<16; i++) {
        printf("%05.2f ", C[i]);
        if((i+1)%4 == 0) printf("\n");
    }

    free(C);

}
