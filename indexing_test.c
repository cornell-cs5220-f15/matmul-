#include <assert.h>
#include <stdio.h>

#include "indexing.h"

void test_cm(void);
void test_rm(void);

int main(void) {
    test_cm();
    test_rm();

    printf("+----------------+\n");
    printf("| ALL TESTS PASS |\n");
    printf("+----------------+\n");
    return 0;
}

// 0 1 2  3
// 4 5 6  7
// 8 9 10 11

void test_cm() {
    const int N = 3;
    const int M = 4;
    int A[] = {
        0, 4, 8,
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
    };

    assert (A[cm(N, M, 0, 0)] == 0);
    assert (A[cm(N, M, 0, 1)] == 1);
    assert (A[cm(N, M, 0, 2)] == 2);
    assert (A[cm(N, M, 0, 3)] == 3);
    assert (A[cm(N, M, 1, 0)] == 4);
    assert (A[cm(N, M, 1, 1)] == 5);
    assert (A[cm(N, M, 1, 2)] == 6);
    assert (A[cm(N, M, 1, 3)] == 7);
    assert (A[cm(N, M, 2, 0)] == 8);
    assert (A[cm(N, M, 2, 1)] == 9);
    assert (A[cm(N, M, 2, 2)] == 10);
    assert (A[cm(N, M, 2, 3)] == 11);
}

void test_rm() {
    const int N = 3;
    const int M = 4;
    int A[] = {
        0, 1, 2,  3,
        4, 5, 6,  7,
        8, 9, 10, 11,
    };

    assert (A[rm(N, M, 0, 0)] == 0);
    assert (A[rm(N, M, 0, 1)] == 1);
    assert (A[rm(N, M, 0, 2)] == 2);
    assert (A[rm(N, M, 0, 3)] == 3);
    assert (A[rm(N, M, 1, 0)] == 4);
    assert (A[rm(N, M, 1, 1)] == 5);
    assert (A[rm(N, M, 1, 2)] == 6);
    assert (A[rm(N, M, 1, 3)] == 7);
    assert (A[rm(N, M, 2, 0)] == 8);
    assert (A[rm(N, M, 2, 1)] == 9);
    assert (A[rm(N, M, 2, 2)] == 10);
    assert (A[rm(N, M, 2, 3)] == 11);
}
