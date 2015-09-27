#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

double *cm_transpose(const double *A, const int lN, const int lM,
                                      const int N,  const int M);
double *rm_transpose(const double *A, const int lN, const int lM,
                                      const int N,  const int M);

#endif // __TRANSPOSE_H__
