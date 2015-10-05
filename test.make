#!/bin/bash

# C and Fortran compilers
CC=icc

# Compiler optimization flags.  You will definitely want to play with these!
OPTFLAGS = #-O3
CFLAGS = -std=gnu99 
FFLAGS = 
LDFLAGS := -fopenmp -nofor_main
LDFLAGS += -axCORE-AVX2


# Add -DDEBUG_RUN to CPPFLAGS to cut down on the cases.
CPPFLAGS = "-DCOMPILER=\"$(CC)\"" "-DFLAGS=\"$(OPTFLAGS)\""

# Compile a C version (using basic_dgemm.c, in this case):
LIBS = -lm -lirng
OBJS = matmul.o

# Libraries and include files for BLAS
LIBMKL=-lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
INCMKL=

vectorize_test: test_vectorize.c
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $(LD_FLAGS) $(INCMKL) $< 

