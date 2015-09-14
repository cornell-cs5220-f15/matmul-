const char* dgemm_desc = "My awesome matmul.";

#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif
/* template by David Bindel,
written by Marc Aurele Gilles

*/



/*
  A,B,C are M-by-M

*/


void row_to_block(const int M, const int nblock, const double *A, double *newA){
	// converts to block indexing and pads the new matrix with zeros so that it is divisble by BLOCK_SIZE
	int bi,bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < BLOCK_SIZE; ++i){
				for(j=0; j < BLOCK_SIZE; ++j){
					if ((bj*BLOCK_SIZE+j) >= M || bi*BLOCK_SIZE+i >= M){ // we can optimize this to delete this "if". this is doing the padding 
					newA[((bj*nblock+bi)*BLOCK_SIZE*BLOCK_SIZE+ i*BLOCK_SIZE+j)]=0;}
					else{
 				newA[(bj*nblock+bi)*BLOCK_SIZE*BLOCK_SIZE+ i*BLOCK_SIZE+j]=A[(j+bj*BLOCK_SIZE)*M+bi*BLOCK_SIZE+i]; }
				
				}
			}
		}
	}
	}


void block_to_row(const int M, const int nblock, double *A, const double *newA){

		
	int bi, bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < BLOCK_SIZE; ++i){
				for(j=0; j < BLOCK_SIZE; ++j){
					if ((bj*BLOCK_SIZE+j)>= M || bi*BLOCK_SIZE+i >= M){ // we can optimize this to delete this "if". this is doing the padding 
					}
					else{
				 A[(j+bj*BLOCK_SIZE)*M+bi*BLOCK_SIZE+i]=newA[(bj*nblock+bi)*BLOCK_SIZE*BLOCK_SIZE+ i*BLOCK_SIZE+j]; }
				
				}
			}
		}
	}
	}	
	
	
	
	

void do_block(const int M, const int nblock,
              const double *A, const double *B, double *C,
              const int bi, const int bj, const int bk)
{
	    int i, j, k;
    for (i = 0; i < BLOCK_SIZE; ++i) {
        for (j = 0; j < BLOCK_SIZE; ++j) {
            double cij = C[((bj*nblock+bi)*BLOCK_SIZE*BLOCK_SIZE+ i*BLOCK_SIZE+j)];
            for (k = 0; k < BLOCK_SIZE; ++k) {
                cij += A[((bk*nblock+bi))*BLOCK_SIZE*BLOCK_SIZE+BLOCK_SIZE*i+k] * B[((bj*nblock)+bk)*BLOCK_SIZE*BLOCK_SIZE+BLOCK_SIZE*k+j];
            }
            C[((bj*nblock+bi)*BLOCK_SIZE*BLOCK_SIZE+ i*BLOCK_SIZE+j)]= cij;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
	/* pad size : size of matrix after padding
		block i= row index of block
		block j = column index of block
		nblock = number of blocks
	*/
	int pad_size, bi, bj, bk, nblock;
	
	if (M%BLOCK_SIZE==0){
		nblock=M/BLOCK_SIZE;
		pad_size=M;}
	else{ pad_size=((M/BLOCK_SIZE)+1)*BLOCK_SIZE;
		nblock=M/BLOCK_SIZE+1;}
		
	double *bA= (double*) malloc(pad_size*pad_size*sizeof(double));
	double *bB= (double*) malloc(pad_size*pad_size*sizeof(double));
	double *bC= (double*) malloc(pad_size*pad_size*sizeof(double));
	// change indexing
	row_to_block(M,nblock, A, bA);
	row_to_block(M,nblock, B, bB);
	row_to_block(M,nblock, C, bC);
	
	
    for (bi = 0; bi < nblock; ++bi) {
        for (bj = 0; bj < nblock; ++bj) {
            for (bk = 0; bk < nblock; ++bk) {
                do_block(M, nblock, bA, bB, bC, bi, bj, bk);
            }
        }
    }
	// reindex back to column
	block_to_row(M,nblock,C,bC);
}
