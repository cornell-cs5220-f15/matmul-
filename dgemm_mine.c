const char* dgemm_desc = "My awesome matmul.";

#include <stdlib.h>

#ifndef L1_BS
#define L1_BS ((int) 16)
#endif

#ifndef L2_BS
#define L2_BS ((int) 8)
#endif

#ifndef L3_BS
#define L3_BS ((int) 10)
#endif


/*

  A,B,C are M-by-M
  L1_BS is the size of sub matrix that will fit into L1
  L2_BS is the number of submatrix that will fit in L2
  L3*L2 is the number of submatrix that will fit in L3

*/
void row_to_block(const int M, const int nblock, const double *restrict A, double *restrict newA)
{
	// converts to block indexing and pads the new matrix with zeros so that it is divisble by L1_BS
	int bi,bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j) >= M || bi*L1_BS+i >= M){ 
						// we can optimize this to delete this "if". this is doing the padding 
						newA[((bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j)]=0;
					}
					else{
						newA[(bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j]=A[(j+bj*L1_BS)*M+bi*L1_BS+i]; 
					}				
				}
			}
		}
	}
}








void block_to_row(const int M, const int nblock, double *restrict A, const double *restrict newA)
{
	int bi, bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j)>= M || bi*L1_BS+i >= M){
					   	// we can optimize this to delete this "if". this is doing the padding 
					}
					else{
						A[(j+bj*L1_BS)*M+bi*L1_BS+i]=newA[(bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j];
				   	}

				}
			}
		}
	}
}	
void row_to_block_transpose(const int M, const int nblock, const double *restrict A, double *restrict newA)
{
	// converts to block indexing and pads the new matrix with zeros so that it is divisble by L1_BS
	int bi,bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j) >= M || bi*L1_BS+i >= M){ 
						// we can optimize this to delete this "if". this is doing the padding 
						newA[((bj*nblock+bi)*L1_BS*L1_BS+ j*L1_BS+i)]=0;
					}
					else{
						newA[(bj*nblock+bi)*L1_BS*L1_BS+ j*L1_BS+i]=A[(j+bj*L1_BS)*M+bi*L1_BS+i]; 
					}				
				}
			}
		}
	}
}






void do_block(const int M, const int nblock,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int bi, const int bj, const int bk)
{
	int i, j, k, BA_A, BA_B, sub_BA_A, sub_BA_B, BA_C, sub_BA_C;
    __assume_aligned(A, 64);
    __assume_aligned(B, 64);
    __assume_aligned(C, 64);

	// BA stands for block adress 
    BA_A=(bk*nblock+bi)*L1_BS*L1_BS;
    BA_B=(bj*nblock+bk)*L1_BS*L1_BS;
    BA_C=(bj*nblock+bi)*L1_BS*L1_BS;
    for (i = 0; i < L1_BS; ++i) {
	// finds sub_BA, tells compiler its aligned
	sub_BA_A=BA_A+L1_BS*i;
	__assume(sub_BA_A%8==0);	
	//same for C                
    	sub_BA_C=BA_C+L1_BS*i;
        __assume(sub_BA_C%8==0);

	for (j = 0; j < L1_BS; ++j){
	    sub_BA_B=BA_B+L1_BS*j;
            __assume(sub_BA_B%8==0);

	    double cij = C[sub_BA_C+j];
	
            for (k = 0; k < L1_BS; ++k) {
		// new kernel using block to row transpose	
                cij += A[sub_BA_A+k] * B[sub_BA_B+k];
       		//without transpose: 
       		//cij += A[((bk*nblock+bi))*L1_BS*L1_BS+L1_BS*i+k] * B[((bj*nblock)+bk)*L1_BS*L1_BS+L1_BS*k+j];
		}
	 C[sub_BA_C+j]= cij;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
	/* pad size : size of matrix after padding
	   block i= row index of block
	   block j = column index of block
	   nblock = number of blocks
	   L2bi, L2bj index of L2
           L2nblock= number of L1block that fit into L2
	   L2rem number of remaining blocks 
	*/
	int pad_size, bi, bj, bk, L2bi, L2bj, L2bk, nblock, L2nblock;

	if (M%L1_BS==0){
		nblock=M/L1_BS;
		pad_size=M;
	}
	else{ 
		pad_size=((M/L1_BS)+1)*L1_BS;
		nblock=M/L1_BS+1;
	}
	
	// number of L2

	if(pad_size%(L2_BS*L1_BS)==0){
	L2nblock=pad_size/(L2_BS*L1_BS);}

	else{L2nblock=pad_size/(L2_BS*L1_BS)+1;}
		
	 //use something like double bA[size] __attribute__((aligned(64))); instead?
	/*
	double bA[pad_size*pad_size] __attribute__((aligned(64))); 
	double bB[pad_size*pad_size] __attribute__((aligned(64))); 
	double bC[pad_size*pad_size] __attribute__((aligned(64))); 
	*/
	
	
	double *restrict bA= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	double *restrict bB= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	double *restrict bC= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	
	
	
	
	// change indexing
	row_to_block(M,nblock, A, bA);
	row_to_block_transpose(M,nblock, B, bB);
	row_to_block(M,nblock, C, bC);

	for (L2bk=0; L2bk < L2nblock; ++L2bk){  
	for (L2bj=0; L2bj < L2nblock; ++L2bj){
	for (L2bi=0; L2bi < L2nblock; ++L2bi){
	for (bk = 0; bk < L2_BS; ++bk) {
		for (bj = 0; bj < L2_BS; ++bj) {
			for (bi = 0; bi < L2_BS; ++bi) {
				if ((L2bi*L2_BS+bi <  nblock) && (L2bj*L2_BS+bj < nblock) && (L2bk*L2_BS +bk <nblock)){
				do_block(M, nblock, bA, bB, bC, L2bi*L2_BS+bi, L2bj*L2_BS+bj, L2bk*L2_BS+bk);}
			}
		}
	}
	}
	}
	}
// reindex back to column
	block_to_row(M,nblock,C,bC);
}
