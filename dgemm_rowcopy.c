#include <stdlib.h>

const char* dgemm_desc = "Three-loop dgemm with a row copy optimization.";

void rowcopy(int M, int row, const double* A, double* A_C)	{
	int j;
	for (j = 0; j < M; ++j)	{
		A_C[j] = A[j*M + row];
	}
}

void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
	int i, j, k;
	double* restrict A_C = (double*) malloc(M*sizeof(double));


	double cij;
	for (i = 0; i < M; ++i) {
		rowcopy(M, i, A, A_C);
		for (j = 0; j < M; ++j) {
			cij = C[j*M+i];
			for (k = 0; k < M; ++k)	{
				cij += A_C[k] * B[j*M+k];
				C[j*M+i] = cij;
			}
		}
	}
	free(A_C);
}
