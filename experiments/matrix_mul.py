import numpy as np
A = np.array([1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16])
B = np.array([2,6,10,14,3,7,11,15,4,8,12,16,5,9,13,17])
C = np.array([0]*16)
for k in range(4):
	for i in range(4):
			C[(k*4):(k*4+4)] += B[k*4+i]*A[i*4:(i*4+4)]

print C

a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
a = np.reshape(a,(4,4))
b = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
b = np.reshape(b,(4,4))
print np.dot(a,b)
