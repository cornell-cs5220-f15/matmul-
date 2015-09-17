#define A(i, j) A[j*M+i]
#define B(i, j) B[j*M+i]
#define C(i, j) C[j*M+i]

const char* dgemm_desc = "Different permutations of loop variables.";

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < M; ++k) {
            double tmp = A(i, k);
            for (j = 0; j < M; ++j)
                C(i, j) += tmp * B(k, j);
        }
    }
}
