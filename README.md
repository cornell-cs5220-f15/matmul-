# Group 6 Matrix Multiplication #
This repository contains the source code for Group 6's implementation of a
double-precision floating point matrix multiplication kernel.

## Kernels ##
After many iterations of optimization and experimentation, we have developed
the following matrix multiplication kernels. Each has its own advantages and
disadvantages. Some are fast on certain inputs; some are slow on certain
inputs. Some are simple; some are complex. Some are tailored for compiler
optimization; some are tailored for minimizing computation. Overall, we've
explored a large number of various tuning tricks and optimizations.

| **Kernel**                                                   | **Description**                                                                                                                        |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| [`dgemm_padded_blocked.c`](dgemm_padded_blocked.c)           | Padded blocked matmul with small fixed-size blocks. `A` gets transposed, `B` gets copied.                                              |
| [`dgemm_padded_if.c`](dgemm_padded_if.c)                     | Padded blocked matmul with small fixed-size blocks. `A` gets transposed, `B` gets copied. If statement avoids superfluous computation. |
| [`dgemm_3_level_blocking.c`](dgemm_3_level_blocking.c)       | Very cache-friendly three-tiered blocked matmul.                                                                                       |
| [`dgemm_3_level_blocking_V2.c`](dgemm_3_level_blocking_V2.c) | Very cache-friendly three-tiered blocked matmul with further tweaks and optimizations.                                                 |
| [`dgemm_big_blocked.c`](dgemm_big_blocked.c)                 | Blocked matmul with large block size. `A` gets transposed in dynamically allocated blocks.                                             |

Additionally, we have a few very simple kernels that we use for reference.

| **Kernel**                               | **Description**                                        |
| ---------------------------------------- | ------------------------------------------------------ |
| [`dgemm_annotated.c`](dgemm_annotated.c) | Naive matmul with compiler annotations (e.g. restrict) |
| [`dgemm_copyopt.c`](dgemm_copyopt.c)     | Naive matmul where `A` is initially transposed.        |

Lastly, we have written numerous experimental kernels along the way that have
been deleted, augmented, or transformed over time. Feel free to explore our
commit history to see the progression of our kernels.

## Report ##
Our midpoint progress report can be found in the [`checkin`](checkin).
Similarly, our final report can be found in the [`report`](report).

## Reproducing Results ##
To build, run, and plot all the kernels in this directory, first log into the
totient cluster. It's important to note that our kernels have only been tuned
for `icc` on the totient cluster. Then, run the following command to build and
submit a job for each kernel.

```bash
make realclean && make run
```

Wait for all the jobs to complete and then run the following command to
generate the plot.

```bash
make plot
```

## Modifying the Builds ##
If you would like to build only certain kernels, you will have to modify
[`Makefile.in.icc`](Makefile.in.icc). In order to reduce the overhead of
writing kernels, we have modified the Makefile to automatically build all
kernels in this directory automatically using the following Makefile logic
found at the top of `Makefile.in.icc`.

```bash
BUILDS := $(wildcard dgemm_*.[cf])          # dgemm_f2c_desc.c, dgemm_basic.c ...
BUILDS := $(basename $(BUILDS))             # dgemm_f2c_desc, dgemm_basic ...
BUILDS := $(subst dgemm_,,$(BUILDS))        # f2c_desc, basic, ...
BUILDS := $(filter-out f2c_desc, $(BUILDS)) # basic, ...
BUILDS := mkl $(BUILDS)                     # mkl, basic, ...
```

If you would like to build a certain set of kernels, simply comment out this
code and replace it with a naive assignment to `BUILDS`. For example, if we
wanted to run the `foo`, `bar`, and `baz`, kernel, then we would modify
`Makefile.in.icc` to look like the following.

```bash
# BUILDS := $(wildcard dgemm_*.[cf])          # dgemm_f2c_desc.c, dgemm_basic.c ...
# BUILDS := $(basename $(BUILDS))             # dgemm_f2c_desc, dgemm_basic ...
# BUILDS := $(subst dgemm_,,$(BUILDS))        # f2c_desc, basic, ...
# BUILDS := $(filter-out f2c_desc, $(BUILDS)) # basic, ...
# BUILDS := mkl $(BUILDS)                     # mkl, basic, ...
BUILDS = foo bar baz
```

## Submitting Jobs ##
The matrix multiplication assignment shipped with a portable batch script for
every kernel that could be submitted to run on the totient cluster. To avoid
the need for multiple batch scripts, we have replaced them with a single
[`runner.pbs`](runner.pbs) scripts that can run an arbitrary kernel.

To run the `foo` kernel, run the following command.

```bash
qsub runner.pbs -N foo -vARG1=matmul-foo
```

Alternatively, you can run the following command:

```bash
make timing-foo.csv
```

## Resources ##
-  **Compiler Optimization**
    - https://software.intel.com/sites/default/files/compiler_qrg12.pdf
    - https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler
    - https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations
    - https://software.intel.com/sites/default/files/Compiler_QRG_2013.pdf
- **Optimization Reports**
    - https://software.intel.com/en-us/articles/vectorization-and-optimization-reports
    - https://software.intel.com/en-us/articles/compilation-of-vectorization-diagnostics-for-intel-c-compiler
- **Vector Instructions**
    - https://software.intel.com/en-us/blogs/2013/avx-512-instructions
    - https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX2&expand=351
