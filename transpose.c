#include "transpose.h"

extern inline double *cm_transpose(const double *A,
                                   const int lN, const int lM,
                                   const int N,  const int M);

extern inline double *rm_transpose(const double *A,
                                   const int lN, const int lM,
                                   const int N,  const int M);
