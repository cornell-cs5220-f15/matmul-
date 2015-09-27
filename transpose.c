#include "transpose.h"

#include <stdlib.h>

#include "indexing.h"

double *cm_transpose(const double *A, const int lN, const int lM,
                                      const int N,  const int M) {
    double *transposed = (double *)malloc(N * M * sizeof(double));
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            transposed[rm(N, M, n, m)] = A[cm(lN, lM, n, m)];
        }
    }
    return transposed;
}

double *rm_transpose(const double *A, const int lN, const int lM,
                                      const int N,  const int M) {
    double *transposed = (double *)malloc(N * M * sizeof(double));
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            transposed[cm(N, M, n, m)] = A[rm(lN, lM, n, m)];
        }
    }
    return transposed;
}
