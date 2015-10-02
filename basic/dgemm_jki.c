const char* dgemm_desc = "k->j->i (outer product).";

#define AT(mat, M, i, j) mat[j*M +i]

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
  int i, j, k;
  for (j = 0; j < M; ++j) 
    {
      for (k = 0; k < M; ++k)
	{
	  /* double cij = AT(C, M, i, j); */
	  double bkj = AT(B,M,k,j);
	  /* double aik = AT(A,M,i,k); */
	  for (i = 0; i < M; ++i) 
	    {
	      /* AT(C, M, i, j) += AT(A,M,i,k) + bkj;  */
	      AT(C,M,i,j) += AT(A,M,i,k)* bkj;
	    }
	}
    }
}
