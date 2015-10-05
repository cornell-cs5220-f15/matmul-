lda = 17

A_INIT = []
num = 0
for i in range(lda):
    A_INIT.append([])
    for j in range(lda):
        A_INIT[i].append(num)
        num += 1

B_INIT = []
C_INIT = []
for i in range(lda):
    B_INIT.append([])
    C_INIT.append([])
    for j in range(lda):
        B_INIT[i].append(num)
        num += 1
        C_INIT[i].append(0)

# transpose and emulate C storage
A_INIT = map(list, zip(*A_INIT))
B_INIT = map(list, zip(*B_INIT))
C_INIT = map(list, zip(*C_INIT))
A = []
B = []
C = []
A_VERIFY = []
B_VERIFY = []
C_VERIFY = []
for j in range(lda):
    for i in range(lda):
        A.append(A_INIT[i][j])
        B.append(B_INIT[i][j])
        C.append(C_INIT[i][j])

        A_VERIFY.append(A_INIT[i][j])
        B_VERIFY.append(B_INIT[i][j])
        C_VERIFY.append(C_INIT[i][j])

BLOCK_SIZE = 11

KERNEL_SIZE = 8
A_KERNEL = [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]]
B_KERNEL = [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]]
C_KERNEL = [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]]

##########################################################################################
##########################################################################################
##########################################################################################
# basic_dgemm
##########################################################################################
##########################################################################################
##########################################################################################
def basic_dgemm(lda, M, N, K, A_START, B_START, C_START):

    row_kernels = M / KERNEL_SIZE if M % KERNEL_SIZE == 0 else (M / KERNEL_SIZE) + 1
    col_kernels = N / KERNEL_SIZE if N % KERNEL_SIZE == 0 else (N / KERNEL_SIZE) + 1
    sli_kernels = K / KERNEL_SIZE if K % KERNEL_SIZE == 0 else (K / KERNEL_SIZE) + 1

    print "[%d, %d, %d]" % (M, N, K)
    print "(%d, %d, %d)" % (row_kernels, col_kernels, sli_kernels)
    
    for bi in range(row_kernels):
        row = bi * KERNEL_SIZE

        M_KERNEL = M-row if row + KERNEL_SIZE > M else KERNEL_SIZE

        for bj in range(col_kernels):
            col = bj * KERNEL_SIZE

            N_KERNEL = N-col if col + KERNEL_SIZE > N else KERNEL_SIZE

            for bk in range(sli_kernels):
                sli = bk * KERNEL_SIZE

                K_KERNEL = K-sli if sli + KERNEL_SIZE > K else KERNEL_SIZE

                print "    (%d, %d, %d)" % (row, col, sli)

                for ki in range(KERNEL_SIZE):
                    ki_row = ki + row
                    ki_sli = ki + sli
                    for kj in range(KERNEL_SIZE):
                        kj_col = kj + col
                        kj_sli = kj + sli

                        # if ki + row >= M or kj + sli >= K:
                        if ki_row >= M or kj_sli >= K:
                            A_KERNEL[ki][kj] = 0
                        else:
                            A_KERNEL[ki][kj] = A[kj_sli*lda + ki_row + A_START]

                        # if ki + sli >= K or kj + col >= N:
                        if ki_sli >= K or kj_col >= N:
                            B_KERNEL[ki][kj] = 0
                        else:
                            B_KERNEL[ki][kj] = B[(kj_col)*lda + ki_sli + B_START]

                        C_KERNEL[ki][kj] = 0
        
                #
                # do multiply
                #
                for iii in range(KERNEL_SIZE):
                    for jjj in range(KERNEL_SIZE):
                        cij = C_KERNEL[iii][jjj]
                        for kkk in range(KERNEL_SIZE):
                            cij += A_KERNEL[iii][kkk] * B_KERNEL[kkk][jjj]
                        C_KERNEL[iii][jjj] = cij

                #
                # copy back to C
                #
                for ki in range(KERNEL_SIZE):
                    ki_row = ki + row
                    # ri_sli = ki + sli
                    for kj in range(KERNEL_SIZE):
                        kj_col = kj + col
                        # rj_sli = kj + sli
                        # if ki + row < M and kj + col < N:
                        if ki_row < M and kj_col < N:
                            C[kj_col*lda + ki_row + C_START] += C_KERNEL[ki][kj]

##########################################################################################
##########################################################################################
##########################################################################################
# do_block
##########################################################################################
##########################################################################################
##########################################################################################
def do_block(lda, i, j, k):
    M = lda-i if i+BLOCK_SIZE > lda else BLOCK_SIZE
    N = lda-j if j+BLOCK_SIZE > lda else BLOCK_SIZE
    K = lda-k if k+BLOCK_SIZE > lda else BLOCK_SIZE

    basic_dgemm(lda, M, N, K, i+k*lda, k+j*lda, i+j*lda)

##########################################################################################
##########################################################################################
##########################################################################################
# square_dgemm
##########################################################################################
##########################################################################################
##########################################################################################
def square_dgemm(M):
    n_blocks = M / BLOCK_SIZE if M % BLOCK_SIZE == 0 else (M / BLOCK_SIZE) + 1

    for bi in range(n_blocks):
        i = bi * BLOCK_SIZE
        for bj in range(n_blocks):
            j = bj * BLOCK_SIZE
            for bk in range(n_blocks):
                k = bk * BLOCK_SIZE

                do_block(M, i, j, k)


square_dgemm(lda)

#
# naive verify
#
for i in range(lda):
    for j in range(lda):
        cij = C_VERIFY[j*lda + i]
        for k in range(lda):
            cij += A_VERIFY[k*lda + i] * B_VERIFY[j*lda + k]
        C_VERIFY[j*lda + i] = cij

indices = [x for x in range(lda*lda)]
bad_indices = []
for i in range(lda):
    for j in range(lda):
        idx = j*lda + i
        if C_VERIFY[idx] != C[idx]:
            print "Fail: index=%d" % idx
            print "  Expected: %d" % C_VERIFY[idx]
            print "       Got: %d" % C[idx]

            bad_indices.append(idx)

bad_indices.sort()

print "Len Indices:    %d" % len(indices)
print "Len BadIndices: %d" % len(bad_indices)

# const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
#     int bi, bj, bk;

#     for (bi = 0; bi < n_blocks; ++bi) {
#         const int i = bi * BLOCK_SIZE;
#         for (bj = 0; bj < n_blocks; ++bj) {
#             const int j = bj * BLOCK_SIZE;
#             for (bk = 0; bk < n_blocks; ++bk) {
#                 const int k = bk * BLOCK_SIZE;

#                 // printf("Block: lda=%d\n       i=%d\n       j=%d\n       k=%d\n", M, i, j, k);

#                 do_block(M, A, B, C, i, j, k);
#             }
#         }
#     }

# M = len(A)

# # const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
# #     int bi, bj, bk;

# #     for (bi = 0; bi < n_blocks; ++bi) {
# #         const int i = bi * BLOCK_SIZE;
# #         for (bj = 0; bj < n_blocks; ++bj) {
# #             const int j = bj * BLOCK_SIZE;
# #             for (bk = 0; bk < n_blocks; ++bk) {
# #                 const int k = bk * BLOCK_SIZE;
# #                 do_block(M, A, B, C, i, j, k);
# #             }
# #         }
# #     }

# n_blocks = M / KERNEL_SIZE
# if M % KERNEL_SIZE != 0:
#     n_blocks += 1

# lda = M

# for bi in range(n_blocks):
#     i = bi * KERNEL_SIZE
#     for bj in range(n_blocks):
#         j = bj * KERNEL_SIZE

#         # execute kernel
#         # const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
#         K_M = lda-i if i+KERNEL_SIZE > lda else KERNEL_SIZE
#         # const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
#         K_N = lda-j if j+KERNEL_SIZE > lda else KERNEL_SIZE
#         # const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

#         print K_M, K_N

#         for kj in range(KERNEL_SIZE):
#             for ki in range(KERNEL_SIZE):
#                 if ki < K_M and kj < K_N:
#                     K[ki][kj] = A[kj+j][ki+i]
#                 else:
#                     K[ki][kj] = 0

#         # # case 1: both are all indices in the matrix
#         # if K_M == KERNEL_SIZE and K_N == KERNEL_SIZE:
#         #     for kj in range(K_N):
#         #         for ki in range(K_M):
#         #                 K[ki][kj] = A[kj+j][ki+i]

#         # elif K_M == KERNEL_SIZE and K_N < KERNEL_SIZE:
#         #     for kj in range(KERNEL_SIZE):
#         #         for ki in range(K_M):
#         #             if kj < K_N:
#         #                 K[ki][kj] = A[kj+j][ki+i]
#         #             else:
#         #                 K[ki][kj] = 0

#         # elif K_M < KERNEL_SIZE and K_N == KERNEL_SIZE:
#         #     for kj in range(K_N):
#         #         for ki in range(KERNEL_SIZE):
#         #             if ki < K_M:
#         #                 K[ki][kj] = A[kj+j][ki+i]
#         #             else:
#         #                 K[ki][kj] = 0

#         # else:# K_M < KERNEL_SIZE and K_N < KERNEL_SIZE
#         #     for kj in range(KERNEL_SIZE):
#         #         for ki in range(KERNEL_SIZE):
#         #             if ki < K_M and kj < K_N:
#         #                 K[ki][kj] = A[kj+j][ki+i]
#         #             else:
#         #                 K[ki][kj] = 0                        

#         for r in K:
#             print r
#         print "- - - -"
