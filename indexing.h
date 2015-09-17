#ifndef __INDEXING_H__
#define __INDEXING_H__

// column-major
double *column_major_at(double *A,
                        const int N,
                        const int i,
                        const int j);

double *column_major_at_blocked(double *A,
                                const int lda,
                                const int i,
                                const int j);

// row-major
double *row_major_at(double *A,
                        const int M,
                        const int i,
                        const int j);

double *row_major_at_blocked(double *A,
                                const int lda,
                                const int i,
                                const int j);

#endif // __INDEXING_H__
