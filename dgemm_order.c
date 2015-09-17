// Automation template test

const char* dgemm_desc = "Order $0$$1$$2$";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{   
    int i, j, k;
    for ($0$ = 0; $0$ < M; ++$0$) {
        for ($1$ = 0; $1$ < M; ++$1$) {
            for ($2$ = 0; $2$ < M; ++$2$) {
                C[j*M+i] += A[k*M+i] * B[j*M+k];
            }
        }
    }
}
