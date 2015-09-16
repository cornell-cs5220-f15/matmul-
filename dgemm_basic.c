const char* dgemm_desc = "Basic, three-loop dgemm.";

#define PERMUTATION 1

void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
#if PERMUTATION == 0
  // i, then j, then k
  int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
#elif PERMUTATION == 1
  // i, then k, then j
#elif PERMUTATION == 2
  // etc.....
#elif PERMUTATION == 3

#elif PERMUTATION == 4

#else // PERUMTATION == 5

#endif
}
