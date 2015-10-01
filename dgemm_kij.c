const char* dgemm_desc = "Loop ordering of kij";

void square_dgemm(const int M, const double *A, const double *B, double *C) {
  int i, j, k;
  for (k = 0; k < M; ++k) {
    for (i = 0; i < M; ++i) {
      for (j = 0; j < M; ++j) {
        double cij = C[j*M+i];
        cij += A[k*M+i] * B[j*M+k];
        C[j*M+i] = cij;
      }
    }
  }
}
