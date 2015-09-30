from sys import argv
import os

if len(argv) == 2:
	suffix = argv[1]

with open("job-" + suffix + ".pbs", "w") as outFile:
	outFile.write("#!/bin/sh -l\n")

	outFile.write("#PBS -l walltime=0:30:00\n")
	outFile.write("#PBS -N matmul-" + suffix + "\n")
	outFile.write("#PBS -j oe\n")

	outFile.write("module load cs5220\n")
	outFile.write("cd $PBS_O_WORKDIR\n")
	outFile.write("./matmul-" + suffix + "\n")

os.system("qsub -l nodes=1:ppn=24 job-" + suffix + ".pbs")
