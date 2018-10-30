import numpy as np
import matplotlib.pyplot as plt

#1 Obtengo los datos de WDBC, sin las primeras columnas(pues son el id del paciente y un string), y solo la segunda columna para después diferenciar.

A=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

A1=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

A2=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(1),dtype="unicode")

#2 Realizo mi implementacion de la matriz de covarianza, la cual comparandola con la que nos da el paquete es bastante similar (la resta de ambas matrices nos da numeros de 10 a la menos 15)



cov=np.zeros((30,30))




for x in range(30):
	for y in range(30):
		cov[x,y]=(np.sum((A[:,x]-np.mean(A[:,x]))*(A[:,y]-np.mean(A[:,y]))))/568.0
		
print ("La matriz de covarianza de los datos es:")
print (cov)




#3 Obtengo los valores y vectores propios usando linalg

w,v = np.linalg.eig(cov)

print ("Los autovalores son: ")
print (w)
print ("Los autovectores son:") 
print (v)

print ("cada autovalor tiene asociado un autovector. Por ejemplo el primer par autovalor/autovector seria "+ str(w[0])+" " + str(v[:,0])) 

#4 

print ("los autovalores que tengan el mayor valor(mayor covarianza), son los componentes principales que estamos buscando (Pues son los que estan mas relacionados). En este caso son " + str(w[0]) + " y " + str(w[1]) + "que tienen los siguientes vectores propios asociados" + str(v[:,0]) + " y " + str(v[:,1]))    


#5 realizo la proyeccion de los valores propios en la matriz original. Y usando los datos de "M" y "B" obtenidos, separamos a los enfermos de los sanos.

Y1= np.dot(A1,v[:,0])
Y2= np.dot(A1,v[:,1])
Y1M=list()
Y2M=list()

Y1B=list()
Y2B=list()

for x in range(len(Y1)):
	if(A2[x]=="M"):
		Y1M.append(Y1[x])
		Y2M.append(Y2[x])
	if(A2[x]=="B"):
		Y1B.append(Y1[x])
		Y2B.append(Y2[x])


plt.figure()
plt.scatter(Y1M,Y2M, c="red")
plt.scatter(Y1B,Y2B, c="blue")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("EstupinanAndres_PCA.pdf")


#6

print("El método de PCA es útil para realizar esta clasificación pues la dimensionalidad de nuestro problema se vio reducida, y los datos muestran una relacion no trivial entre las componentes principales.(pues la poblacion enferma y no enferma se encuentra en regiones cercanas en el plot)") 















