const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
#endif
#define ALIGN __attribute__ ((aligned (32)))


void MMult4by4VRegAC(int M, const double * restrict A, const double * restrict B, double * restrict C)
{
  int p;
  __m256d a0,b0,b1,b2,b3,c00,c01,c02,c03;
  __m256d ab0,ab1,ab2,ab3;
  c00 = _mm256_load_pd(C+0);
  c01 = _mm256_load_pd(C+M);
  c02 = _mm256_load_pd(C+2*M);
  c03 = _mm256_load_pd(C+3*M);

  for (p = 0; p <M; p++){
    a0 = _mm256_load_pd(A+4*p);
    b0 = _mm256_broadcast_sd(B+p);
    b1 = _mm256_broadcast_sd(B+p+M);
    b2 = _mm256_broadcast_sd(B+p+2*M);
    b3 = _mm256_broadcast_sd(B+p+3*M);
    ab0 = _mm256_mul_pd(a0,b0);
    ab1 = _mm256_mul_pd(a0,b1);
    ab2 = _mm256_mul_pd(a0,b2);
    ab3 = _mm256_mul_pd(a0,b3);
    c00 = _mm256_add_pd(c00,ab0);
    c01 = _mm256_add_pd(c01,ab1);
    c02 = _mm256_add_pd(c02,ab2);
    c03 = _mm256_add_pd(c03,ab3);
  }
  _mm256_store_pd(C+0,c00);
  _mm256_store_pd(C+M,c01);
  _mm256_store_pd(C+2*M,c02);
  _mm256_store_pd(C+3*M,c03);
}




void ContigA(int lda, int i, const double * restrict a, double * restrict acont)
{
  int j,k;
  k = 0;
  for (j = 0; j < lda; j++)
    {
      acont[k] = a[j*lda+i];
      acont[k+1] = a[j*lda+i+1];
      acont[k+2] = a[j*lda+i+2];
      acont[k+3] = a[j*lda+i+3];
      k += 4;
    }
}


void to_kdgemm(int M, int Mk,
	       const double * restrict A,
	       double * restrict Ak,
	       const double * restrict B,
	       double * restrict Bk,
	       double * restrict C,
	       double * restrict Ck)

{
    for (int i = 0; i < M; i ++)
        for (int j = 0; j < M; j++) {
            Ak[i*Mk + j] = A[i*M + j];
            Bk[i*Mk + j] = B[i*M + j];
            Ck[i*Mk + j] = 0.0;
        }
    for (int i = 0; i < Mk; i ++)
        for (int j = M; j < Mk; j++) {
            Ak[i*Mk + j] = 0.0;
            Bk[i*Mk + j] = 0.0;
            Ck[i*Mk + j] = 0.0;
        }
    for (int i = M; i < Mk; i ++)
        for (int j = 0; j < Mk; j++) {
            Ak[i*Mk + j] = 0.0;
            Bk[i*Mk + j] = 0.0;
            Ck[i*Mk + j] = 0.0;
        }
}


void from_kdgemm(int M, int Mk,
                 double * restrict A,
                 const double * restrict Ak)
{
    for (int i = 0; i < M; i ++)
        for (int j = 0; j < M; j++) {
            A[i*M + j] = Ak[i*Mk + j];
        }
}


void square_dgemm(int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    int i,j;
    const int n_4 = M / 4 + (M%4? 1 : 0);
    const int Mk = n_4*4;
    double* Ak = _mm_malloc(Mk * Mk * sizeof(double), 32);
    double* Bk = _mm_malloc(Mk * Mk * sizeof(double), 32);
    double* Ck = _mm_malloc(Mk * Mk * sizeof(double), 32);
    to_kdgemm(M, Mk, A, Ak, B, Bk, C, Ck);

    double* LocA = _mm_malloc(4 * Mk * sizeof(double), 32);
    for (i=0; i<Mk; i+=4){
      ContigA(Mk,i,Ak,LocA);
      for(j=0; j<Mk; j+=4) {
    	MMult4by4VRegAC (Mk, LocA, Bk + j*Mk, Ck + i + j*Mk);
      }
    }
    _mm_free(LocA);

    from_kdgemm(M, Mk, C, Ck);

    _mm_free(Ck);
    _mm_free(Bk);
    _mm_free(Ak);
}

