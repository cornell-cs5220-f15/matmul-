#include <stdlib.h>
#include <stdio.h>

const char* dgemm_desc = "row transpose kij";

// copies and transposes A
// would in-memory transpose be faster/ more space efficient?
void transpose_array(const int M, const double *A, double *copied)
{
	int row, column;

	for (column = 0; column < M; ++column){
		for (row = 0; row < M; ++row){
			copied[(row * M) + column] = A[(column * M) + row];
		}
	}
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
   	double* a_transposed = (double*) malloc(M * M * sizeof(double));

   	// create a transposed A in row-major order
   	transpose_array(M, A, a_transposed);

    // write to C in row-major order
    for (k = 0; k < M; ++k) {
      for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
        	double cij = C[j*M+i];

        	// we can compute these before the inner loop so two less multiplication per cycle!
        	int a_start = i*M;
        	int b_start = j*M;

        	cij += a_transposed[a_start + k] * B[b_start + k];

          C[j*M+i] = cij;
        }
      }
    }

    free(a_transposed);
}
