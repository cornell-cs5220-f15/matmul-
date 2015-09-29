block_list = [16,32,64,128,256,512]

sourcename   = 'dgemm_mine.c'
makefilename = 'Makefile.in.icc'
jobname      = 'blocked'

### Create the source files with new blocking
for block in block_list:
	targetname = 'dgemm_block'+str(block)+'.c'
	with open(sourcename,'r') as sourcefile:
		with open(targetname,'w') as targetfile:
			for line in sourcefile:
				if line.find('#define BLOCK_SIZE') > -1 :
					targetfile.write('#define BLOCK_SIZE ((int) '+str(block)+')\n')
				else:
					targetfile.write(line)

### Create a new Makefile
with open(makefilename,'r') as sourcefile:
	with open('Makefile.in.blocking','w') as targetfile:
		for line in sourcefile:
			if line.find('BUILDS=') == 0 :
				buildstr = ''
				for block in block_list:
					buildstr = buildstr+ 'block'+str(block)+' '
				targetfile.write('BUILDS='+buildstr+'\n')
			else:
				targetfile.write(line)

### Create new pbs scripts
for block in block_list:
	with open('job-'+jobname+'.pbs','r') as sourcefile:
		targetname = 'job-block'+str(block)+'.pbs'
		with open(targetname,'w') as targetfile:
			for line in sourcefile:
				if line.find(jobname) > -1:
					targetfile.write(line.replace('jobname','block'+str(block)))
				else:
					targetfile.write(line)
