import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq


#1


A=np.genfromtxt("signal.dat", delimiter=",")
B=np.genfromtxt("incompletos.dat")


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
dt=Ax[1]-Ax[2]
s=0.0
for b in range(n):

	m.append(s)

	s=0.0
	for k in range(n):
		s=s+(Ay[k]*np.exp((-1j)*2.0*np.pi*k*(b/n)))
	
#4




#fft_x= fft(Ay)
freq= fftfreq(n,dt)
#plt.plot(freq,abs(fft_x))
#plt.show()
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


plt.figure()
plt.plot(freq,np.abs(m))
plt.show()
#para ver la tranasformada filtrada

filtrada= np.fft.ifft(m)

plt.figure()
plt.plot(Ax, filtrada)
plt.plot(Ax, Ay)
plt.show()




	












