#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "copy.h"
#include "indexing.h"

void test_cm_copy(void);
void test_rm_copy(void);
void test_cm_copy_into(void);

int main(void) {
    test_cm_copy();
    test_rm_copy();

    printf("+----------------+\n");
    printf("| ALL TESTS PASS |\n");
    printf("+----------------+\n");
    return 0;
}

void test_cm_copy() {
    const int lN = 5;
    const int lM = 6;
    const int N = 3;
    const int M = 4;
    double A[] = {
        99, 99, 99, 99, 99,
        99, 0,  4,  8,  99,
        99, 1,  5,  9,  99,
        99, 2,  6,  10, 99,
        99, 3,  7,  11, 99,
        99, 99, 99, 99, 99,
    };

    double *transposed = cm_copy(&A[cm(lN, lM, 1, 1)], lN, lM, N, M);

    assert (transposed[cm(N, M, 0, 0)] == 0);
    assert (transposed[cm(N, M, 0, 1)] == 1);
    assert (transposed[cm(N, M, 0, 2)] == 2);
    assert (transposed[cm(N, M, 0, 3)] == 3);
    assert (transposed[cm(N, M, 1, 0)] == 4);
    assert (transposed[cm(N, M, 1, 1)] == 5);
    assert (transposed[cm(N, M, 1, 2)] == 6);
    assert (transposed[cm(N, M, 1, 3)] == 7);
    assert (transposed[cm(N, M, 2, 0)] == 8);
    assert (transposed[cm(N, M, 2, 1)] == 9);
    assert (transposed[cm(N, M, 2, 2)] == 10);
    assert (transposed[cm(N, M, 2, 3)] == 11);

    free(transposed);
}

void test_rm_copy() {
    const int lN = 5;
    const int lM = 6;
    const int N = 3;
    const int M = 4;
    double A[] = {
        99, 99, 99, 99, 99, 99,
        99, 0,  1,  2,  3,  99,
        99, 4,  5,  6,  7,  99,
        99, 8,  9,  10, 11, 99,
        99, 99, 99, 99, 99, 99,
    };

    double *transposed = rm_copy(&A[rm(lN, lM, 1, 1)], lN, lM, N, M);

    assert (transposed[rm(N, M, 0, 0)] == 0);
    assert (transposed[rm(N, M, 0, 1)] == 1);
    assert (transposed[rm(N, M, 0, 2)] == 2);
    assert (transposed[rm(N, M, 0, 3)] == 3);
    assert (transposed[rm(N, M, 1, 0)] == 4);
    assert (transposed[rm(N, M, 1, 1)] == 5);
    assert (transposed[rm(N, M, 1, 2)] == 6);
    assert (transposed[rm(N, M, 1, 3)] == 7);
    assert (transposed[rm(N, M, 2, 0)] == 8);
    assert (transposed[rm(N, M, 2, 1)] == 9);
    assert (transposed[rm(N, M, 2, 2)] == 10);
    assert (transposed[rm(N, M, 2, 3)] == 11);

    free(transposed);
}

void test_cm_copy_into() {
    const int lN = 5;
    const int lM = 6;
    const int N = 3;
    const int M = 4;
    double A[] = {
        99, 99, 99, 99, 99,
        99, 0,  4,  8,  99,
        99, 1,  5,  9,  99,
        99, 2,  6,  10, 99,
        99, 3,  7,  11, 99,
        99, 99, 99, 99, 99,
    };

    const int A_N = 10;
    const int A_M = 20;
    double A_[A_N * A_M];
    memset(A_, 0, A_N * A_M);

    cm_copy_into(&A[cm(lN, lM, 1, 1)], lN, lM, N, M, A_, A_N, A_M);

    assert (A_[cm(A_N, A_M, 0, 0)] == 0);
    assert (A_[cm(A_N, A_M, 0, 1)] == 1);
    assert (A_[cm(A_N, A_M, 0, 2)] == 2);
    assert (A_[cm(A_N, A_M, 0, 3)] == 3);
    assert (A_[cm(A_N, A_M, 1, 0)] == 4);
    assert (A_[cm(A_N, A_M, 1, 1)] == 5);
    assert (A_[cm(A_N, A_M, 1, 2)] == 6);
    assert (A_[cm(A_N, A_M, 1, 3)] == 7);
    assert (A_[cm(A_N, A_M, 2, 0)] == 8);
    assert (A_[cm(A_N, A_M, 2, 1)] == 9);
    assert (A_[cm(A_N, A_M, 2, 2)] == 10);
    assert (A_[cm(A_N, A_M, 2, 3)] == 11);
}
