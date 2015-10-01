# Group 6 Matrix Multiplication #
This repository contains the source code for Group 6's implementation of a
double-precision floating point matrix multiplication kernel.

## Kernels ##
| **Kernel**                   | **Description**                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `dgemm_big_blocked.c`        | Blocked matmul with large block size. `A` gets transposed in dynamically allocated blocks.                       |
| `dgemm_padded_blocked.c`     | Padded blocked matmul with small fixed-size blocks. `A` gets transposed, `B` gets copied.                        |
| `dgemm_3_level_blocking.c`   | Three-tiered blocked matmul.                                                                                     |
| `dgemm_annotated.c`          | Naive matmul with compiler annotations (e.g. restrict)                                                           |
| `dgemm_basic.c`              | Naive matmul                                                                                                     |
| `dgemm_blas.c`               | BLAS matmul.                                                                                                     |
| `dgemm_blocked.c`            | Blocked matmul.                                                                                                  |
| `dgemm_compiler.c`           |                                                                                                                  |
| `dgemm_copyopt.c`            | Naive matmul where `A` is initially transposed.                                                                  |
| `dgemm_f2c.f`                | Fortran matmul                                                                                                   |

## Resources ##
-  **Compiler Optimization**
    - https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler
    - https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations
    - https://software.intel.com/sites/default/files/Compiler_QRG_2013.pdf
