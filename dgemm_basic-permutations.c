const char* dgemm_desc = "Basic loop dgemm permutations";

// more convenient access
#define A(i, j) A[j*M + i]
#define B(i, j) B[j*M + i]
#define C(i, j) C[j*M + i]

#define PERMUTATION 3

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    int i, j, k;
#if PERMUTATION == 0
    // i, then j, then k
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double c_ij = C(i, j);
            for (k = 0; k < M; ++k) {
                c_ij += A(i, k) * B(k, j);
            }
            C(i, j) = c_ij;
        }
    }
#elif PERMUTATION == 1
    // i, then k, then j
    for(i = 0; i < M; ++i) {
        for(k = 0; k < M; ++k) {
            double a_ik = A(i, k);
            for(j = 0; j < M; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
#elif PERMUTATION == 2
    // j, then i, then k
    for(j = 0; j < M; ++j) {
        for(i = 0; i < M; ++i) {
            double c_ij = C(i, j);
            for(k = 0; k < M; ++k) {
                c_ij += A(i, k) * B(k, j);
            }
            C(i, j) = c_ij;
        }
    }

#elif PERMUTATION == 3
    // j, then k, then i
    for(j = 0; j < M; ++j) {
        for(k = 0; k < M; ++k) {
            double b_kj = B(k, j);
            for(i = 0; i < M; ++i) {
                C(i, j) += A(i, k) * b_kj;
            }
        }
    }
#elif PERMUTATION == 4
    // k, then i, then j
    for(k = 0; k < M; ++k) {
        for(i = 0; i < M; ++i) {
            double a_ik = A(i, k);
            for(j = 0; j < M; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
#else // PERUMTATION == 5
    // k, then j, then i
    for(k = 0; k < M; ++k) {
        for(j = 0; j < M; ++j) {
            double b_kj = B(k, j);
            for(i = 0; i < M; ++i) {
                C(i, j) += A(i, k) * b_kj;
            }
        }
    }
#endif
}
