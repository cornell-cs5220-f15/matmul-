################################################################################
# matrix pretty printing
################################################################################
def matrix_to_filename(c, n, m):
    return "mats/{}-{}x{}.tex".format(c, n, m)

def matrix_transpose_to_filename(c, n, m):
    return "mats/{}'-{}x{}.tex".format(c, n, m)

def element_to_string(c, row, col):
    return "{}_{{{}, {}}}".format(c, row, col)

def row_to_string(v):
    return " & ".join(str(x) for x in v) + r" \\"

def matrix_to_string(m):
    return "\n".join([
        r"\begin{pmatrix}",
        "\n".join(row_to_string(v) for v in m),
        r"\end{pmatrix}",
    ])

def matrix_to_file(m, filename):
    with open(filename, "w") as f:
        f.write(matrix_to_string(m))

################################################################################
# matrix construction
################################################################################
def empty(n, m):
    return [["" for j in range(m)] for i in range(n)]

def nbym(c, n, m):
    return [[element_to_string(c, i, j) for j in range(m)] for i in range(n)]

def transpose(xs):
    return [list(col) for col in zip(*xs)]

################################################################################
# convenience
################################################################################
def nbym_to_file(c, n, m):
    matrix_to_file(nbym(c, n, m), matrix_to_filename(c, n, m))

def nbym_transpose_to_file(c, n, m):
    matrix_to_file(transpose(nbym(c, n, m)), matrix_transpose_to_filename(c, n, m))

def ijk(c, N, K, M):
    i = 0
    j = 0
    count = 0
    A = empty(N, K)
    B = empty(K, M)
    C = empty(N, M)

    for j in range(M):
        for k in range(K):
            count += 1
            if A[i][k] == "":
                A[i][k] = str(count)
            B[k][j] = str(count)
        C[i][j] = str(count)

    return A, B, C

################################################################################
# main
################################################################################
def main():
    for i in range(1, 11):
        for j in range(1, 11):
            for c in ["a", "b", "c"]:
                nbym_to_file(c, i, j)
                nbym_transpose_to_file(c, i, j)

if __name__ == "__main__":
    main()
