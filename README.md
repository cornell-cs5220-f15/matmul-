# Group 6 Matrix Multiplication #
This repository contains the source code for Group 6's implementation of a
double-precision floating point matrix multiplication kernel.

## Kernels ##
| **Kernel**                   | **Description**                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `dgemm_big_blocked.c`        | Blocked multiplication. `A` gets transposed in blocks. Large block size.               |
| `dgemm_padded_blocked.c`     | Padded blocked multiplication. `A` gets transposed, `B` gets copied. Small block size. |
| `dgemm_annotated.c`          |                                                                                        |
| `dgemm_basic.c`              |                                                                                        |
| `dgemm_blas.c`               |                                                                                        |
| `dgemm_blocked.c`            |                                                                                        |
| `dgemm_compiler.c`           |                                                                                        |
| `dgemm_copyopt.c`            |                                                                                        |
| `dgemm_copyopt_transpose.c`  |                                                                                        |
| `dgemm_f2c_desc.c`           |                                                                                        |
| `dgemm_mine.c`               |                                                                                        |
| `dgemm_mine_ian.c`           |                                                                                        |
| `dgemm_mine_ian_SSE_types.c` |                                                                                        |
