import matplotlib.pyplot as plt

max_cores = 128
serial_fraction = 0.1

cores = range(1,max_cores+1)
speedup = range(0,max_cores) #temporary
for i in cores:
	speedup[i-1] = 1.0/(serial_fraction + (1.0-serial_fraction)/i)

plt.plot(cores,speedup)
plt.xlabel('Number of cores')
plt.ylabel('Idealized speedup')
plt.show() 
	


