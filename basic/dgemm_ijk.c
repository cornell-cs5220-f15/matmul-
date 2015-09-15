const char* dgemm_desc = "k->j->i (outer product).";

#define AT(mat, M, i, j) mat[j*M +i]

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  int i, j, k;
  for (i = 0; i < M; ++i) 
    {
      for (j = 0; j < M; ++j) 
	{
	  double cij = AT(C, M, i, j);
	  /* double bkj = AT(B,M,k,j);  */
	  for (k = 0; k < M; ++k)
	    {
	      /* AT(C, M, i, j) += AT(A,M,i,k) + bkj;  */
	      cij += AT(A,M,i,k) * AT(B,M,k,j);
	    }
	  AT(C,M,i,j) = cij;
	}
    }
}
