import numpy as np
import matplotlib.pyplot as plt

#1

A=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

A1=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30))
#2

Ap=np.transpose(A)

cov=np.zeros((30,30))



for x in range(30):
	for y in range(30):
		cov[x,y]=(np.sum((A[:,x]-np.mean(A[:,x]))*(A[:,y]-np.mean(A[:,y]))))/568.0
		
print ("La matriz de covarianza de los datos es:")
print (cov)


#3

w,v = np.linalg.eig(cov)

print ("Los autovalores son: ")
print (w)
print ("Los autovectores son:") 
print (v)

print ("cada autovalor tiene asociado un autovector mostrado anteriormente en orden. Por ejemplo el primer par autovalor/autovector seria "+ str(w[0])+" " + str(v[0])) 

print ("los autovalores que tengan el mayor valor(mayor covarianza), son los componentes principales que estamos buscando (Pues son los que estan mas relacionados). En este caso son " + str(w[0]) + " y " + str(w[1]) + "que tienen los siguientes vectores propios asociados" + str(v[0]) + " y " + str(v[1]))    


PM= np.hstack((v[0], v[1]))
PM2=[v[0],v[1]]
PM3= np.transpose(PM2)

print ("matriz proyeccion")

print (PM)
print (PM2)
print (PM3)

Y= np.dot(A1,PM3)

mn = []
mns = plt.scatter(x=Y[:,0],y=Y[:,1])
mn.append(mns)

plt.show()




















