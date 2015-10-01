#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clear.h"
#include "indexing.h"

void test_cm_clear_but(void);
void test_rm_clear_but(void);

int main(void) {
    test_cm_clear_but();
    test_rm_clear_but();

    printf("+----------------+\n");
    printf("| ALL TESTS PASS |\n");
    printf("+----------------+\n");
    return 0;
}

void test_cm_clear_but(void) {
    const int lN = 8;
    const int lM = 6;
    const int N = 5;
    const int M = 4;
    double A[] = {
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    };


    cm_clear_but(A, lN, lM, N, M);

    double expected[] = {
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        0,  0,  0,  0,  0,  0, 0, 0,
        0,  0,  0,  0,  0,  0, 0, 0,
    };

    assert (0 == memcmp(A, expected, lN * lM));
}

void test_rm_clear_but(void) {
    const int lN = 6;
    const int lM = 8;
    const int N = 4;
    const int M = 5;
    double A[] = {
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    };


    rm_clear_but(A, lN, lM, N, M);

    double expected[] = {
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        99, 99, 99, 99, 99, 0, 0, 0,
        0,  0,  0,  0,  0,  0, 0, 0,
        0,  0,  0,  0,  0,  0, 0, 0,
    };

    assert (0 == memcmp(A, expected, lN * lM));
}
