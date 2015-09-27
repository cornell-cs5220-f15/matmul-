#include "indexing.h"

int cm(const int N, const int M, const int i, const int j) {
    return i + N*j;
}

// row-major
int rm(const int N, const int M, const int i, const int j) {
    return i*M + j;
}

