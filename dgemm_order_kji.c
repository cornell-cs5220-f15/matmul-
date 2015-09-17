const char* dgemm_desc = "Order kji";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{   
    int i, j, k;
    for (k = 0; k < M; ++k) {
        for (j = 0; j < M; ++j) {
            for (i = 0; i < M; ++i)
                C[j*M+i] += A[k*M+i] * B[j*M+k];
        }
    }
}
