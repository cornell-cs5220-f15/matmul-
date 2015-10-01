const char* dgemm_desc = "My awesome dgemm.";
#include <stdlib.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#define BLOCK_SIZE_HALF ((int) 32)
#endif

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE

typedef union
{
  __m128d v;
  double d[2];
} d2v;
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{
	
	d2v* restrict avec = (d2v*) malloc(BLOCK_SIZE * BLOCK_SIZE_HALF * sizeof(d2v));
	d2v* restrict bvec = (d2v*) malloc(BLOCK_SIZE * BLOCK_SIZE_HALF * sizeof(d2v));
	int i, j, k;
	int ood = K%2; 
	for(k=0;k<K;++k){//smallA = A'
		int wb=k%2;
		for(i=0;i<M;++i){
			avec[i*BLOCK_SIZE_HALF+k/2].d[wb]=A[k*lda+i];
		}
	}
	if(ood){
		//printf("%d\n",K);
		for(i=0;i<M;++i){
			avec[i*BLOCK_SIZE_HALF+(K-1)/2].d[1]=0;
		}
	}

	for(j=0;j<N;++j){
		for(k=0;k<K-1;k+=2){
			bvec[j*BLOCK_SIZE_HALF+k/2].d[0]=B[j*lda+k];
			bvec[j*BLOCK_SIZE_HALF+k/2].d[1]=B[j*lda+k+1];
		}
	}
	
	if(ood){
		for(j=0;j<N;++j){
			bvec[j*BLOCK_SIZE_HALF+(K-1)/2].d[0]=B[j*lda+K-1];
			bvec[j*BLOCK_SIZE_HALF+(K-1)/2].d[1]=0;
		}
	}

	__assume_aligned(avec,512);
	__assume_aligned(bvec,512);
    
    for (j = 0; j < N; ++j) {
         for (i = 0; i < M; ++i){
            double cij = C[j*lda+i];
			d2v cvec;
			cvec.v =_mm_setzero_pd();;
            for (k = 0; k < K; k+=2) {
				#pragma vector always 
                cvec.v += avec[i*BLOCK_SIZE_HALF+k/2].v * bvec[j*BLOCK_SIZE_HALF+k/2].v;
            }
            C[j*lda+i] = cij+cvec.d[0]+cvec.d[1];
        }
    }

	free(avec);
	free(bvec);

}

void do_block(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}

