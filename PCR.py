import numpy as np
#import matplotlib.pyplot as plt

#1

A=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

#2

Ap=np.transpose(A)

cov=np.zeros((30,30))



for x in range(30):
	for y in range(30):
		cov[x,y]=(np.sum((A[:,x]-np.mean(A[:,x]))*(A[:,y]-np.mean(A[:,y]))))/568.0
		
print "La matriz de covarianza de los datos es:"
print cov


#3

w,v = np.linalg.eig(cov)

print "Los autovectores y autovalores son: "
print w
print v

