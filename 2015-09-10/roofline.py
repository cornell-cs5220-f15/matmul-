import matplotlib.pyplot as plt

## compute node specs
# resolution = 200
# min_AI     = 0.0
# max_AI     = 16.0
# memory_bandwidth = 59
# peak_flop        = 12.0*16*2.4

## compute node + 2 coprocessors
resolution  = 200
min_AI      = 0.0
max_AI      = 16.0
memory_bandwidth = 699
peak_flop        = 12.0*16*2.4 + 2.0*60.0*16.0*1.053

AI = [0.0]*resolution
flop_rate = [0.0]*resolution
for i in range(0,resolution):
	AI[i] = min_AI+(max_AI-min_AI)*i/resolution
	flop_rate[i] = min(AI[i]*memory_bandwidth,peak_flop)

plt.plot(AI,flop_rate)
plt.xlabel('Arithmetic intensity')
plt.ylabel('GFlops/s')
## For the compute node
# plt.title('Roofline for the compute node')
# plt.savefig('roofline_compute.png')
## For the compute node + 2 coprocessors
plt.title('Roofline for node + 2 Phi boards')
plt.savefig('roofline_compute2phi.png')