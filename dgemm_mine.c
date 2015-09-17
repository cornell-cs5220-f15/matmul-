const char* dgemm_desc = "My awesome dgemm.";

<<<<<<< Updated upstream
#define PERMUTATION 5

inline void naive_transpose(int M, double* mat) {
    int c, r;
    double tmp;
    for(int c = 0; c < M; ++c) {
        for(int r = 0; r < M; ++r) {
            tmp = mat[c*M+r];
            mat[c*M+r] = mat[r*M+c];
            mat[r*M+c] = tmp;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    // why did you give us const?  aren't we supposed to play with this?
    double* bad_idea = (double *)(&(*A));
    naive_transpose(M, bad_idea);
#if PERMUTATION == 0
=======
void square_dgemm(const int M, const double *A, const double *B, double *C) {
>>>>>>> Stashed changes
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
<<<<<<< Updated upstream
#elif PERMUTATION == 1
    // i, then k, then j
    int i, j, k;
    for(i = 0; i < M; ++i) {
        for(k = 0; k < M; ++k) {
            double aki = A[k*M+i];
            for(j = 0; j < M; ++j) {
                C[j*M+i] += aki * B[j*M+k];
            }
        }
    }
#elif PERMUTATION == 2
    // j, then i, then k
    int i, j, k;
    for(j = 0; j < M; ++j) {
        for(i = 0; i < M; ++i) {
            double cij = C[j*M+i];
            for(k = 0; k < M; ++k) {
                cij += A[k*M+i] * B[j*M+k];
            }
            C[j*M+i] = cij;
        }
    }

#elif PERMUTATION == 3
    // j, then k, then i
    int i, j, k;
    for(j = 0; j < M; ++j) {
        for(k = 0; k < M; ++k) {
            double bjk = B[j*M+k];
            for(i = 0; i < M; ++i) {
                C[j*M+i] += A[k*M+i] * bjk;
            }
        }
    }
#elif PERMUTATION == 4
    // k, then i, then j
    int i, j, k;
    for(k = 0; k < M; ++k) {
        for(i = 0; i < M; ++i) {
            double aki = A[k*M+i];
            for(j = 0; j < M; ++j) {
                C[j*M+i] += aki * B[j*M+k];
            }
        }
    }
#else // PERUMTATION == 5
    // k, then j, then i
    int i, j, k;
    for(k = 0; k < M; ++k) {
        for(j = 0; j < M; ++j) {
            double bjk = B[j*M+k];
            for(i = 0; i < M; ++i) {
                C[j*M+i] += A[k*M+i] * bjk;
            }
        }
    }
#endif
    naive_transpose(M, bad_idea);
=======
>>>>>>> Stashed changes
}

