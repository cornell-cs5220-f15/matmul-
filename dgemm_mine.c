const char* dgemm_desc = "My awesome dgemm.";

#define PERMUTATION 5

inline void naive_transpose(int M, double* mat) {
    int c, r;
    double tmp;
    for(int c = 0; c < M; ++c) {
        for(int r = 0; r < M; ++r) {
            tmp = mat[c*M+r];
            mat[c*M+r] = mat[r*M+c];
            mat[r*M+c] = tmp;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    // why did you give us const?  aren't we supposed to play with this?
    double* bad_idea = (double *)(&(*A));
    naive_transpose(M, bad_idea);
    
    
    
    
    naive_transpose(M, bad_idea);
}

