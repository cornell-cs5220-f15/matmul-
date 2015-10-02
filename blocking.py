
# A = [[0,  6, 12, 18, 24, 30],
#      [1,  7, 13, 19, 25, 31],
#      [2,  8, 14, 20, 26, 32],
#      [3,  9, 15, 21, 27, 33],
#      [4, 10, 16, 22, 28, 34],
#      [5, 11, 17, 23, 29, 25]]

# A = [[0,  6, 12, 18],
#      [1,  7, 13, 19],
#      [2,  8, 14, 20],
#      [3,  9, 15, 21]]

# A = [[0,  6, 12],
#      [1,  7, 13],
#      [2,  8, 14]]

A = [[0,  6, 12, 18, 24, 30, 8],
     [1,  7, 13, 19, 25, 31, 8],
     [2,  8, 14, 20, 26, 32, 8],
     [3,  9, 15, 21, 27, 33, 8],
     [4, 10, 16, 22, 28, 34, 8],
     [5, 11, 17, 23, 29, 25, 8],
     [8,  8,  8,  8,  8,  8, 8]]

KERNEL_SIZE = 4
K = [[0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0]]

M = len(A)

# const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
#     int bi, bj, bk;

#     for (bi = 0; bi < n_blocks; ++bi) {
#         const int i = bi * BLOCK_SIZE;
#         for (bj = 0; bj < n_blocks; ++bj) {
#             const int j = bj * BLOCK_SIZE;
#             for (bk = 0; bk < n_blocks; ++bk) {
#                 const int k = bk * BLOCK_SIZE;
#                 do_block(M, A, B, C, i, j, k);
#             }
#         }
#     }

n_blocks = M / KERNEL_SIZE
if M % KERNEL_SIZE != 0:
    n_blocks += 1

lda = M

for bi in range(n_blocks):
    i = bi * KERNEL_SIZE
    for bj in range(n_blocks):
        j = bj * KERNEL_SIZE

        # execute kernel
        # const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
        K_M = lda-i if i+KERNEL_SIZE > lda else KERNEL_SIZE
        # const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
        K_N = lda-j if j+KERNEL_SIZE > lda else KERNEL_SIZE
        # const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

        print K_M, K_N

        for kj in range(KERNEL_SIZE):
            for ki in range(KERNEL_SIZE):
                if ki < K_M and kj < K_N:
                    K[ki][kj] = A[kj+j][ki+i]
                else:
                    K[ki][kj] = 0

        # # case 1: both are all indices in the matrix
        # if K_M == KERNEL_SIZE and K_N == KERNEL_SIZE:
        #     for kj in range(K_N):
        #         for ki in range(K_M):
        #                 K[ki][kj] = A[kj+j][ki+i]

        # elif K_M == KERNEL_SIZE and K_N < KERNEL_SIZE:
        #     for kj in range(KERNEL_SIZE):
        #         for ki in range(K_M):
        #             if kj < K_N:
        #                 K[ki][kj] = A[kj+j][ki+i]
        #             else:
        #                 K[ki][kj] = 0

        # elif K_M < KERNEL_SIZE and K_N == KERNEL_SIZE:
        #     for kj in range(K_N):
        #         for ki in range(KERNEL_SIZE):
        #             if ki < K_M:
        #                 K[ki][kj] = A[kj+j][ki+i]
        #             else:
        #                 K[ki][kj] = 0

        # else:# K_M < KERNEL_SIZE and K_N < KERNEL_SIZE
        #     for kj in range(KERNEL_SIZE):
        #         for ki in range(KERNEL_SIZE):
        #             if ki < K_M and kj < K_N:
        #                 K[ki][kj] = A[kj+j][ki+i]
        #             else:
        #                 K[ki][kj] = 0                        

        for r in K:
            print r
        print "- - - -"
