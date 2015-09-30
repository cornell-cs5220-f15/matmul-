#include "transpose.h"

extern inline double *cm_transpose(const double *A,
                                   const int lN, const int lM,
                                   const int N,  const int M);

extern inline double *rm_transpose(const double *A,
                                   const int lN, const int lM,
                                   const int N,  const int M);

extern inline void *cm_transpose_into(const double *A,
                                      const int lN, const int lM,
                                      const int N,  const int M,
                                      double *A_,
                                      const int A_N,  const int A_M);
