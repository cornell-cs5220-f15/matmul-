const char* dgemm_desc = "I can't do this anymore";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  int i, j, k;

  for (i = 0; i < M; i += 2) {
    for (j = 0; j < M; j += 2) {
      double acc00 = 0;
      double acc01 = 0;
      double acc10 = 0;
      double acc11 = 0;

      for (k = 0; k < M; k++) {
        acc00 += A[j+k*M]     * B[k+i*M];
        acc01 += A[j+(k+1)*M] * B[k+i*M];
        acc10 += A[j+k*M]     * B[k+1+i*M];
        acc11 += A[j+(k+1)*M] * B[k+1+i*M];
      }

      C[j+i*M] = acc00;
      C[j+(i+1)*M] = acc01;
      C[j+1+i*M] = acc10;
      C[j+1+(i+1)*M] = acc11;
    }
  }
}
