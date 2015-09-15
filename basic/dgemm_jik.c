const char* dgemm_desc = "jik";

#define AT(mat, M, i, j) mat[j*M +i]

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  int i, j, k;
  for (j = 0; j < M; ++j) 
    {
      for (i = 0; i < M; ++i) 
	{
	  /* double cij = C[j*M+i]; */
	  double cij = AT(C, M, i, j);
	    for (k = 0; k < M; ++k)
	    {
	      /* cij += A[k*M+i] * B[j*M+k]; */
	      /* C[j*M+i] = cij; */
	      cij += AT(A,M,i,k) * AT(B,M,k,j);
	    }
	    AT(C,M,i,j) = cij;
	}
    }
}
