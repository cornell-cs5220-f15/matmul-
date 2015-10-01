# Group 6 Matrix Multiplication #
This repository contains the source code for Group 6's implementation of a
double-precision floating point matrix multiplication kernel.

## Kernels ##
After many iterations of optimization and experimentation, we have developed
the following matrix multiplication kernels. Each has its own advantages and
disadvantages. Some are fast on certain inputs, some are slow on certain
inputs, some are simple, some are complex, etc.

| **Kernel**                    | **Description**                                                                                                                        |
| ----------------------------  | -------------------------------------------------------------------------------------------------------------------------------------- |
| `dgemm_padded_blocked.c`      | Padded blocked matmul with small fixed-size blocks. `A` gets transposed, `B` gets copied.                                              |
| `dgemm_padded_if.c`           | Padded blocked matmul with small fixed-size blocks. `A` gets transposed, `B` gets copied. If statement avoids superfluous computation. |
| `dgemm_3_level_blocking.c`    | Three-tiered blocked matmul.                                                                                                           |
| `dgemm_3_level_blocking_V2.c` | Three-tiered blocked matmul with further optimizations.                                                                                |
| `dgemm_big_blocked.c`         | Blocked matmul with large block size. `A` gets transposed in dynamically allocated blocks.                                             |

Additionally, we have a few very simple kernels that we use for reference.

| **Kernel**          | **Description**                                        |
| ------------------- | ------------------------------------------------------ |
| `dgemm_annotated.c` | Naive matmul with compiler annotations (e.g. restrict) |
| `dgemm_copyopt.c`   | Naive matmul where `A` is initially transposed.        |

## Resources ##
-  **Compiler Optimization**
    - https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler
    - https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations
    - https://software.intel.com/sites/default/files/Compiler_QRG_2013.pdf
