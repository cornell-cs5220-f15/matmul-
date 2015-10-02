const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void kernel_dgemm(const int lda, const int M, const int N, const int K,
                  double *A,  double *B, double *C)
{
  int i,j,k;
  double *cj1, *cj2, *cj3, *cj4, *bj1, *bj2, *bj3, *bj4;
  __m256d a0,a1,a2,a3,b0,b1,b2,b3,c00,c01,c02,c03;
  __m256d b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
  double *ak1, *ak2, *ak3, *ak4;

  for (j = 0; j < N; j+=4){
    cj1 = C + j*lda;
    cj2 = cj1 + lda;
    cj3 = cj2 + lda;
    cj4 = cj3 + lda;
    bj1 = B + j*lda;
    bj2 = bj1 + lda;
    bj3 = bj2 + lda;
    bj4 = bj3 + lda;
    for (k = 0; k < K; k+=4) {
      b0 = _mm256_broadcast_sd(bj1++);
      b1 = _mm256_broadcast_sd(bj2++);
      b2 = _mm256_broadcast_sd(bj3++);
      b3 = _mm256_broadcast_sd(bj4++);
      
      b4 = _mm256_broadcast_sd(bj1++);
      b5 = _mm256_broadcast_sd(bj2++);
      b6 = _mm256_broadcast_sd(bj3++);
      b7 = _mm256_broadcast_sd(bj4++);
      
      b8 = _mm256_broadcast_sd(bj1++);
      b9 = _mm256_broadcast_sd(bj2++);
      b10 = _mm256_broadcast_sd(bj3++);
      b11 = _mm256_broadcast_sd(bj4++);
      
      b12 = _mm256_broadcast_sd(bj1++);
      b13 = _mm256_broadcast_sd(bj2++);
      b14 = _mm256_broadcast_sd(bj3++);
      b15 = _mm256_broadcast_sd(bj4++);

      ak1 =  A  + k*lda;
      ak2 =  ak1 + lda;
      ak3 =  ak2 + lda;
      ak4 =  ak3 + lda;
      for ( i =0; i < M; i+=4) {
  	a0 = _mm256_load_pd(ak1+i);
  	a1 = _mm256_load_pd(ak2+i);
  	a2 = _mm256_load_pd(ak3+i);
  	a3 = _mm256_load_pd(ak4+i);
	
  	c00 = _mm256_load_pd(cj1+i);
  	c01 = _mm256_load_pd(cj2+i);
  	c02 = _mm256_load_pd(cj3+i);
  	c03 = _mm256_load_pd(cj4+i);
	
  	c00 = _mm256_fmadd_pd(a0,b0,c00);
  	c01 = _mm256_fmadd_pd(a0,b1,c01);
  	c02 = _mm256_fmadd_pd(a0,b2,c02);
  	c03 = _mm256_fmadd_pd(a0,b3,c03);
	
  	c00 = _mm256_fmadd_pd(a1,b4,c00);
  	c01 = _mm256_fmadd_pd(a1,b5,c01);
  	c02 = _mm256_fmadd_pd(a1,b6,c02);
  	c03 = _mm256_fmadd_pd(a1,b7,c03);
	
  	c00 = _mm256_fmadd_pd(a2,b8,c00);
  	c01 = _mm256_fmadd_pd(a2,b9,c01);
  	c02 = _mm256_fmadd_pd(a2,b10,c02);
  	c03 = _mm256_fmadd_pd(a2,b11,c03);
	
  	c00 = _mm256_fmadd_pd(a3,b12,c00);
  	c01 = _mm256_fmadd_pd(a3,b13,c01);
  	c02 = _mm256_fmadd_pd(a3,b14,c02);
  	c03 = _mm256_fmadd_pd(a3,b15,c03);
	
  	_mm256_store_pd(cj1 + i,c00);
  	_mm256_store_pd(cj2 + i,c01);
  	_mm256_store_pd(cj3 + i,c02);
  	_mm256_store_pd(cj4 + i,c03);
      }
    }
  }
}

void to_dgemm_B(int lda, int Nb, int Kb, int Kk, const double * restrict btot, double * restrict bcont)
{
  for (int j =0; j < Nb; j++) {
    for (int k =0; k < Kb; k++) {
      bcont[Kk*j+k] = btot[lda*j+k];
    }
  }
}

void to_dgemm_A(int lda, int Kb, int Mb, int Mk, const double * restrict atot, double * restrict acont)
{
  for (int k =0; k < Kb; k++) {
    for (int i =0; i < Mb; i++) {
      acont[Mk*k+i] = atot[lda*k+i];
    }
  }
}

void from_dgemm_C(int lda, int Nb, int Mb, int Mk, double * restrict ctot, double * restrict ccont)
{
  for (int j =0; j < Nb; j++) {
    for (int i =0; i < Mb; i++) {
      ctot[lda*j+i] += ccont[Mk*j+i];
    }
  }
}



void square_dgemm(int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    int i,j,k;
    __m256d a0,a1,a2,a3,b0,b1,b2,b3,c00,c01,c02,c03;
    __m256d b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
    const int n_4 = M / 4 + (M%4? 1 : 0);
    const int Msize = n_4*4;
    double* Ak = _mm_malloc(Msize * Msize * sizeof(double), 32);
    double* Bk = _mm_malloc(Msize * Msize * sizeof(double), 32);
    double* Ck = _mm_malloc(Msize * Msize * sizeof(double), 32);
    memset(Ak, 0, Msize * Msize * sizeof(double));
    memset(Bk, 0, Msize * Msize * sizeof(double));
    memset(Ck, 0, Msize * Msize * sizeof(double));
    to_dgemm_A(M,M,M,Msize,A,Ak);
    to_dgemm_B(M,M,M,Msize,B,Bk);
    kernel_dgemm(Msize,Msize,Msize,Msize, Ak,Bk,Ck);
    from_dgemm_C(M,M,M,Msize,C,Ck);
    _mm_free(Ck);
    _mm_free(Bk);
    _mm_free(Ak);
}

