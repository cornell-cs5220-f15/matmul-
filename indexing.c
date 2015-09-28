#include "indexing.h"

double *column_major_at(double *A,
                        const int N,
                        const int i,
                        const int j) {
    return &A[i + j*N];
}

double *column_major_at_blocked(double *A,
                                const int lda,
                                const int i,
                                const int j) {
    return column_major_at(A, lda, i, j);
}

double *row_major_at(double *A,
                        const int M,
                        const int i,
                        const int j) {
    return &A[i*M + j];
}

double *row_major_at_blocked(double *A,
                                const int lda,
                                const int i,
                                const int j) {
    return row_major_at(A, lda, i, j);
}
