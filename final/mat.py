def row(xs):
    return " & ".join(str(x) for x in xs) + r" \\"

def matrix(m):
    return "\n".join([
        r"\begin{pmatrix}",
        "\n".join(row(v) for v in m),
        r"\end{pmatrix}",
    ])

def nbym(n, m, c):
    xs = [["{}_{{{}, {}}}".format(c, i, j) for j in range(m)] for i in range(n)]
    return matrix(xs)

def nbym_to_file(n, m, c):
    with open("mats/{}-{}x{}.tex".format(c, n, m), "w") as f:
        f.write(nbym(n, m ,c))

def main():
    for i in range(1, 11):
        for j in range(1, 11):
            for c in ["a", "b", "c"]:
                nbym_to_file(i, j, c)

if __name__ == "__main__":
    main()
