import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
import scipy as sc
from scipy import interpolate

#1


A=np.genfromtxt("signal.dat", delimiter=",")
B=np.genfromtxt("incompletos.dat", delimiter=",")


#2

Ax=A[:,0]
Ay=A[:,1]

plt.figure()
plt.plot(Ax,Ay, label= "datos de signal.dat")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("EstupinanAndres_signal.pdf")	


#3

n=len(Ax)
m=list()
dt=Ax[2]-Ax[1]
s=0.0
for b in range(n):

	m.append(s)

	s=0.0
	for k in range(n):
		s=s+(Ay[k]*np.exp((-1j)*2.0*np.pi*k*(float(b)/float(n))))
	
#4




#fft_x= fft(Ay)
#plt.figure()
freq= fftfreq(n,dt)
#plt.plot(freq,(fft_x))
#plt.savefig("transfourierO.pdf")
#Solo para comparar
plt.figure()
plt.plot(freq,np.abs(m), label="transformada de fourier de los datos")
plt.xlabel("Frecuencias")
plt.ylabel("F(f(t))")
plt.legend()
plt.savefig("EstupinanAndres_TF.pdf")

#5

print ("las frecuencias principales de la señal son los picos mas altos de la grafica. En ese caso son los picos que superan el valor de 500 en el eje y, y estan cerca al 0 de frecuencias")

#6

for x in range(len(freq)):
	if(abs(freq[x])>1000):
		m[x]=0
#for x in range(len(freq)):
#	if(abs(freq[x])>1000):
#		fft_x[x]=0

#plt.figure()
#plt.plot(freq,np.abs(m))
#plt.show()
#para ver la transformada filtrada

filtrada= np.fft.ifft(m)
#filtrada2= np.fft.ifft(fft_x)

#plt.figure()
#plt.plot(Ax, np.real(filtrada2)+np.imag(filtrada2), label="Signal filtrada2")
#plt.plot(Ax, Ay)
#plt.savefig("EstupinanAndres_filtrada2.pdf")


plt.figure()
plt.plot(Ax, np.real(filtrada)+np.imag(filtrada), label="Signal filtrada")
plt.plot(Ax, Ay)
plt.savefig("EstupinanAndres_filtrada.pdf")

#7

print (" No se puede hacer la transformada de fourier de datos incompletos, pues la tr")


	
#8

B1=B[:,0]
B2=B[:,1]

f1=interp1d(B1,B2, kind="quadratic")
f2=interp1d(B1,B2, kind="cubic")

plt.figure()
plt.plot(B1,B2)
#plt.plot(B1,f1)
#plt.plot(B1,f2)
plt.show()

#print (f1)

X=np.linspace(0,0.028515625,1000)
def cuadraticsplines(A,B,X):
	cuadraticBspline = sc.interpolate.splrep(A, B, k=2)
	return sc.interpolate.splev(X ,cuadraticBspline)


def cubicsplines(A,B,X):
	cubicBspline = sc.interpolate.splrep(A, B, k=3)
	return sc.interpolate.splev(X ,cubicBspline)

plt.figure()
plt.scatter(B1,B2)
plt.plot(X, cuadraticsplines(B1,B2,X))
plt.plot(X, cubicsplines(B1,B2,X))
plt.legend(['Cuadratic inter','Cubic inter', 'Puntos'])
plt.savefig("EstupinanAndres_Interpola.pdf")































