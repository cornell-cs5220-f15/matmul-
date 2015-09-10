import matplotlib.pyplot as plt
import math

peak_flop_rate   = 153600000000
memory_bandwidth = 25600000000
L3_size          = 6*pow(2,20)

N = range(1,L3_size/8)
flops = []
for n in N:
	if n < 72:
		flops.append( n / 12.0 * memory_bandwidth)
	elif n <= 512:
		flops.append( peak_flop_rate)
	elif n <= L3_size/16.0:
		flops.append( memory_bandwidth/4.0)
	else:
		flops.append( memory_bandwidth/8.0)

plt.plot(N,flops)
plt.xlabel('Problem Size N')
plt.ylabel('Flops per second')
plt.show() 
