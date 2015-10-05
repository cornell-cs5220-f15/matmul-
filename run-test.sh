#!/bin/bash

make -f test.make
qsub -l nodes=1:ppn=24 job-test.pbs
