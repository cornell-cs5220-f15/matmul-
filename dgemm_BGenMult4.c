const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 60)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void kernel_dgemm(const int lda, const int Mk, const int Nk, const int Kk,
                  double *A,  double *B, double *C)
{
  int i,j,k;
  double *cj1, *cj2, *cj3, *cj4, *bj1, *bj2, *bj3, *bj4;
  __m256d a0,a1,a2,a3,b0,b1,b2,b3,c00,c01,c02,c03;
  __m256d b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
  double *ak1, *ak2, *ak3, *ak4;

  for (j = 0; j < Nk; j+=4){
    cj1 = C + j*lda;
    cj2 = cj1 + lda;
    cj3 = cj2 + lda;
    cj4 = cj3 + lda;
    bj1 = B + j*lda;
    bj2 = bj1 + lda;
    bj3 = bj2 + lda;
    bj4 = bj3 + lda;
    for (k = 0; k < Kk; k+=4) {
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
      for ( i =0; i < Mk; i+=4) {
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


void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void to_kdgemm(int M,
	       const double * restrict A,
	       double * restrict Ak,
	       const double * restrict B,
	       double * restrict Bk)
{
    for (int i = 0; i < M*M; i ++) {
            Ak[i] = A[i];
	    Bk[i] = B[i];
        }
}

void from_kdgemm(int M,
	       double * restrict C,
		 double * restrict Ck)
{
    for (int i = 0; i < M*M; i ++) {
            C[i] = Ck[i];
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




void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int b4mult = (BLOCK_SIZE / 4 + (BLOCK_SIZE%4? 1 : 0))*4;
    const int n_blocks = M / b4mult + (M%b4mult? 1 : 0);
    const int Msize = n_blocks*b4mult;
    double* Ak = _mm_malloc(Msize * Msize * sizeof(double), 32);
    double* Bk = _mm_malloc(Msize * Msize * sizeof(double), 32);
    double* Ck = _mm_malloc(Msize * Msize * sizeof(double), 32);
    memset(Ak, 0, Msize * Msize * sizeof(double));
    memset(Bk, 0, Msize * Msize * sizeof(double));
    memset(Ck, 0, Msize * Msize * sizeof(double));
    to_dgemm_A(M,M,M,Msize,A,Ak);
    to_dgemm_B(M,M,M,Msize,B,Bk);
    int bi, bj, bk;
    double *ak, *aki, *bjk;
    double *cj, *cji;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int jj = bj * b4mult;
	cj = Ck + (jj)*Msize;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int kk = bk * b4mult;
	    bjk = Bk + (jj)*Msize + kk;
	    ak = Ak + kk*Msize;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int ii = bi * b4mult;
		cji = cj + ii;
		aki = ak + ii;
		kernel_dgemm(Msize,b4mult,b4mult, b4mult, aki, bjk, cji);
            }
        }
    }
    from_dgemm_C(M,M,M,Msize,C,Ck);
    _mm_free(Ak);
    _mm_free(Bk);
    _mm_free(Ck);
}

