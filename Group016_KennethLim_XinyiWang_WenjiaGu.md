---
id: group016
members: Kenneth Lim, Xinyi Wang, Wenjia Gu
netids: kl545, xw327, wg233
---

Overview
=======

At time of submission, we’ve tried some combination of the following things:

- Modification of compiler flags
- Loop ordering/unrolling
- Blocking
- Memory Alignment
- Array Transpose

We have work-in-progress, or intend to try:

- Microkernels optimized for multiple block sizes, with block size partitioning targeting different cache levels
- More explicit OpenMP parallelization, e.g. pragma parallel blocks

# Modification of Compiler Flags
This is the lowest hanging fruit. We use ICC for compilation with the following flags:

```shell
-O3 -fast -ftree-vectorize
-opt-prefetch -unroll-aggressive
-parallel -openmp
-ansi-alias -restrict
-xHost -axCORE-AV2
```
In brief, we let the compiler do most of the heavy lifting in terms of loop unrolling, vectorization, parallelizing the code, and skipping redundant locality checks. We also compile exclusively for Intel’s Haswell architecture in accordance with the specifications of the cluster.

Using good compiler flags boosted performance noticeably regardless of algorithmic implementation. In particular, the Fortran implementation appears to be particularly sensitive to compile-time optimizations. As is, it is stable at around 10GFlop/s, but can be tuned such that it approaches a peak performance of almost 70GFlop/s. One does not simply beat Fortran. Or Blas. Or MKL.

Some cases (normally our own code) appeared to perform better when `-O2` was used instead of `-O3`. Almost all of them turned out to be the consequence of bad design on our part --- the compiler was unable to effectively vectorize or unroll the loops when more aggressive optimization was attempted. In these cases, we benefitted from adding the diagnostic flags:

```shell
-qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout
```

which returned information about loop peeling, stride length, and vectorization during the compile process.

# Loop Ordering and Unrolling

We attempted loop unrolling (because we think we can do better than the compiler at menial tasks). As a starting point, one sees a good opportunity in `basic_dgemm` of the provided `dgemm_blocked.c`, which is a simple kernel that handles the arithmetic for the block operation. Since only `BLOCK_SIZE` number of operations will be performed and the number is often small, it seems feasible to hard-code every operation. However, we did not notice any difference in performance between the unrolled version and the unmodified version. It is likely that the compiler already handles this chunk well enough behind the scenes, and that we will have to consider using inlined assembly instructions if we are to obtain any discernible performance gains.

Loop ordering yielded good results. In a naive kernel (be it `dgemm_basic.c`, `basic_dgemm` in `dgemm_blocked.c`, or the equivalent in subsequent blocking implementations), the default `i`, `j`, `k`, from outer to inner loop is not efficient because the matrices are stored in column-major. In this order, `i` must stride through a number of elements equal to the dimension of the matrix on every iteration. In contrast, utilizing the `j`, `k`, `i` order increases performance because `j` changes less frequently, and `i` exploits unit stride in the fastest innermost loop.

![Loop Ordering](timing-loop-order.png)

# Array Transpose

Transposing the matrix before multiplication allows us to go from finding a matrix-matrix product, to a matrix-vector product combined with a vector-vector product. Doing this in a straightforward manner increased the performance of naive DGEMM beyond the simple blocked DGEMM. A caveat here is that while in the previous section we elucidated the optimal loop ordering to be `j`, `k`, `i`, the optimal ordering here is in fact `i`, `j`, `k` because we are now striding in row order. In particular, one can nitpick and minimize the size of the innermost loop by caching the components of the vector-vector product outside the loop. It is also interesting to note that when the input size crosses 1024x1024, `j`, `i`, `k` dominates. We suspect that this is because transposing the array past this point creates a vector that is significantly longer than the stride, negating the performance gains previously described.

![Array Transpose](timing-transpose.png)

# Memory Alignment

We modified `matmul.c` to allocate the inputs to `square_dgemm` such that they are aligned to 16 byte boundaries in memory. This allows us to assert in `square_dgemm` that the inputs are aligned, and have the compiler skip these checks at runtime. This produces a modest speed improvement at smaller input dimensions, but seems to be detrimental as the dimension size increases. However, as we progress into a more involved implementation of the blocked approach, in which matrix sizes are kept reasonably small, we expect that the benefits from memory alignment will be magnified.

![16-byte Boundary Alignment](timing-aligned-16.png)
16-byte Boundary Alignment

![64-byte Boundary Alignment](timing-aligned-64.png)
64-byte Boundary Alignment

# Blocking

Starting from `dgemm_blocked.c`, we experimented with block sizes of 8, 16, 32, and 64. In all cases, there was no discernible difference in performance. However, with a modified version of the blocking approach applying the transpose operation detailed in the foregoing paragraphs, we find that a block size of 32 is optimal. This is due to the fact that each compute node has a 256kb 8-way associative cache. Using a block size of 32 allows blocks from A, B, and C to fit comfortably within the cache with wiggle room for the computational overheads. The following graph also represents our current best result. Moving forward, we expect to focus on this area with an emphasis on creating a multi-level block approach to target the L2 and L1 cache (avoiding L3 since it is shared amongst all cores and can easily be invalidated).

![Current Best](timing-current-best.png)

# Playing with OpenMP

As an aside, it should be noted that we're currently not giving the compiler any hints on loop parallelization. OpenMP is running in single-threaded mode, whereas it may be possible to improve performance if we are able to design the code such that the loop containing the critical section can be executed in parallel. OpenBLAS and MKL seem to make heavy use of synchronization in order to get to their levels of performance. Time permitting, this would be worthwhile trying!
