const char* dgemm_desc = "My mixed blocked dgemm with order J-K-j-I-k-i.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 48)
#endif

void square_dgemm(const int lda, const double *A, const double *B, double *C)
{
    const int n_blocks = lda / BLOCK_SIZE + (lda%BLOCK_SIZE? 1 : 0);
    int bi = 0, bj = 0, bk = 0;
    int Bi = 0, Bj = 0, Bk = 0;
    int i, j, k;
    double *As, *Bs, *Cs;
    while (bj < n_blocks) {
        const int N = (Bj+BLOCK_SIZE > lda? lda-Bj : BLOCK_SIZE);
        bk = 0;
        Bk = 0;
        while (bk < n_blocks) {
            const int K = (Bk+BLOCK_SIZE > lda? lda-Bk : BLOCK_SIZE);
            Bs = B + Bk + Bj*lda;
            for (j = 0; j < N; ++j){
                bi = 0;
                Bi = 0;
                while (bi < n_blocks) {
                    const int M = (Bi+BLOCK_SIZE > lda? lda-Bi : BLOCK_SIZE);
                    As = A + Bi + Bk*lda;
                    Cs = C + Bi + Bj*lda;
                    
                    double cj[BLOCK_SIZE];
                    for (i = 0; i < M; ++i){
                        cj[i] = Cs[j*lda+i];
                    }
                    for (k = 0; k < K; ++k) {
                        for (i = 0; i < M; ++i) {
                            cj[i] += As[k*lda+i] * Bs[j*lda+k];
                        }
                    }
                    for (i = 0; i < M; ++i){
                        Cs[j*lda+i] = cj[i];
                    }
                    ++bi;
                    Bi +=  BLOCK_SIZE;
                }
            }
            ++bk;
            Bk +=  BLOCK_SIZE;
        }
        ++bj;
        Bj += BLOCK_SIZE;
    }
}

