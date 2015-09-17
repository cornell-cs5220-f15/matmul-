const char* dgemm_desc = "Order ikj";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{   
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < M; ++k) {
//            double cij = C[j*M+i];
            for (j = 0; j < M; ++j)
                C[j*M+i] += A[k*M+i] * B[j*M+k];
//            C[j*M+i] = cij;
        }
    }
}
