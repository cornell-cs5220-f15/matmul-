//========================================================================
// dgemm_loop.c
//========================================================================
// DGEMM blocked implementation with alternate loop ordering. Of the six
// possible loop orderings between i/j/k (output row, output column,
// partial product index), our experiments showed that the most amenable
// to the column-major layout of the matrices was the jki ordering. The
// idea is that we want to reuse the data within a cache line (i.e.,
// portion of the current column) as much as possible before accessing
// another cache line (i.e., changing to the next column) in order to
// maximize both temporal and spatial locality. Given that, we want the
// innermost loop (with the least amount of time between iterations) to
// iterate across i (i.e., elements in a given column of the
// output). For the next innermost loop, choosing to iterate across k
// allows us to only have to change the active column of A, whereas
// choosing to iterate across j would mean we would have to change the
// active column of both B and C.

const char* dgemm_desc = "Blocked dgemm with alternate loop ordering.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            for (i = 0; i < M; ++i) {
                // One caveat here is that we lose the ability to
                // accumulate multiple partial products using local
                // registers for a given output element since we need to
                // store back to the output matrix for every iteration
                // here.
                double cij = C[j*lda+i];
                cij += A[k*lda+i] * B[j*lda+k];
                C[j*lda+i] = cij;
            }
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

void square_dgemm(const int M, const double *A, const double *B, double *C)
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
