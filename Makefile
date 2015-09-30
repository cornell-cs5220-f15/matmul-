# ---
# Platform-dependent configuration
#
# If you have multiple platform-dependent configuration options that you want
# to play with, you can put them in an appropriately-named Makefile.in.
# For example, the default setup has a Makefile.in.icc and Makefile.in.gcc.

PLATFORM=icc

include Makefile.in.$(PLATFORM)
DRIVERS=$(addprefix matmul-,$(BUILDS))
TIMINGS=$(addsuffix .csv,$(addprefix timing-,$(BUILDS)))

.PHONY:	all
all:	$(DRIVERS)

# ---
# Rules to build the drivers

matmul-%: $(OBJS) dgemm_%.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-f2c: $(OBJS) dgemm_f2c.o dgemm_f2c_desc.o fdgemm.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-blas: $(OBJS) dgemm_blas.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBBLAS)

matmul-mkl: $(OBJS) dgemm_mkl.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBMKL)

matmul-veclib: $(OBJS) dgemm_veclib.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) -framework Accelerate

matmul-compiler: $(OBJS) dgemm_compiler.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

# ---
# Rules to build the tests

tests: indexing_test transpose_test copy_test

indexing_test: indexing_test.c indexing.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

transpose_test: transpose_test.c transpose.o indexing.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

copy_test: copy_test.c copy.o indexing.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

# --
# Rules to build object files

matmul.o: matmul.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $(EXPERIMENTAL_OPT_FLAGS) $(PGO_FLAG) $(CPPFLAGS) $<

%.o: %.f
	$(FC) -c $(FFLAGS) $(OPTFLAGS) $<

dgemm_blas.o: dgemm_blas.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCBLAS) $<

dgemm_mkl.o: dgemm_blas.c
	$(CC) -o $@ -c $(CFLAGS) $(CPPFLAGS) $(INCMKL) $<

dgemm_veclib.o: dgemm_blas.c
	clang -o $@ -c $(CFLAGS) $(CPPFLAGS) -DOSX_ACCELERATE $<

dgemm_big_blocked_%.o: dgemm_big_blocked.c
	$(CC) -o $@ -c $(CFLAGS) $(OPTFLAGS) $(EXPERIMENTAL_OPT_FLAGS) $(PGO_FLAG) $(CPPFLAGS) $< -DBLOCK_SIZE=$*

dgemm_padded_blocked_%.o: dgemm_padded_blocked.c
	$(CC) -o $@ -c $(CFLAGS) $(OPTFLAGS) $(EXPERIMENTAL_OPT_FLAGS) $(PGO_FLAG) $(CPPFLAGS) $< -DBLOCK_SIZE=$*

# ---
# Rules for building timing CSV outputs

.PHONY: run run-local
run:    $(TIMINGS)

run-local:
	( for build in $(BUILDS) ; do ./matmul-$$build ; done )

timing-%.csv: matmul-%
	qsub runner.pbs -N $* -vARG1=$<

# ---
#  Rules for plotting

.PHONY: plot
plot:
	python plotter.py $(BUILDS)

# ---
#  Rules for cleaning

.PHONY:	clean realclean
clean:
	rm -f matmul-* *.o

realclean: clean
	rm -f *~ timing-*.csv timing.pdf

# ---
#  Rules for printing

print-%: ; @echo $*=$($*)

