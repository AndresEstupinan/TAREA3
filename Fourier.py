import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
import scipy as sc
from scipy import interpolate

#1 Almacenamos los datos de signal y incompletos


A=np.genfromtxt("signal.dat", delimiter=",")
B=np.genfromtxt("incompletos.dat", delimiter=",")


#2 Graficamos

Ax=A[:,0]
Ay=A[:,1]

plt.figure()
plt.plot(Ax,Ay, label= "datos de signal.dat")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("EstupinanAndres_signal.pdf")	


#3 Usamos la implementacion propia para fourier 

n=len(Ax)
m=list()
dt=Ax[2]-Ax[1]
s=0.0
for b in range(n):

	m.append(s)

	s=0.0
	for k in range(n):
		s=s+(Ay[k]*np.exp((-1j)*2.0*np.pi*k*(float(b)/float(n))))
	

#4 Graficamos usando el paquete fftfreq




mp= fft(Ay)
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

#5 Frecuencias principales

print ("las frecuencias principales de la signal son los picos mas altos de la grafica. En ese caso son los picos que superan el valor de 500 en el eje y, y estan cerca al 0 de frecuencias")

#6 Realizamos el filtro

for x in range(len(freq)):
	if(abs(freq[x])>1000):
		m[x]=0


#plt.figure()
#plt.plot(freq,np.abs(m))
#plt.show()
#para ver la transformada filtrada, solo para comparar

filtrada= np.fft.ifft(m)
#filtrada2= np.fft.ifft(fft_x)

#plt.figure()
#plt.plot(Ax, np.real(filtrada2)+np.imag(filtrada2), label="Signal filtrada2")
#plt.plot(Ax, Ay)
#plt.savefig("EstupinanAndres_filtrada2.pdf")
#solo para comparar


plt.figure()
plt.plot(Ax, np.real(filtrada)+np.imag(filtrada), label="Signal filtrada")
plt.savefig("EstupinanAndres_filtrada.pdf")

#7

print (" No se puede hacer la transformada de fourier de datos incompletos, pues el fin de realizar tal transformada es transformar una funcion que depende del tiempo a una que dependa de las frecuencias. Si los datos se encuentran incompletos, no se puede saber con presicion las frecuencias en cada instante de tiempo y por lo tanto la implementacion que estamos realizando para la transformada serian varios picos no despreciables de frecuencias donde se pierde la informacion de la funcion original. ")


	
#8 Interpolacion usando cubic y quadratic splines (para que se ajusten mejor a la funcion deseada)

B1=B[:,0]
B2=B[:,1]



X=np.linspace(0,0.028515625,512)
def cuadraticsplines(A,B,X):
	cuadraticBspline = sc.interpolate.splrep(A, B, k=2)
	return sc.interpolate.splev(X ,cuadraticBspline)


def cubicsplines(A,B,X):
	cubicBspline = sc.interpolate.splrep(A, B, k=3)
	return sc.interpolate.splev(X ,cubicBspline)

#plt.figure()
#plt.scatter(B1,B2)
#plt.plot(X, cuadraticsplines(B1,B2,X))
#plt.plot(X, cubicsplines(B1,B2,X))
#plt.legend(['Cuadratic inter','Cubic inter', 'Puntos'])
#plt.savefig("EstupinanAndres_Interpola.pdf")
#para ver las interpolaciones

X11=cuadraticsplines(B1,B2,X)
X22=cubicsplines(B1,B2,X)



#9

FBAD=fft(B2)
nBAD=len(B1)
dtBAD=B1[1]-B1[0]
freqBAD= fftfreq(nBAD,dtBAD)


F11=fft(X11)
F22=fft(X22)
nGOOD=512
dtGOOD= X[1]-X[0]
freqGOOD= fftfreq(nGOOD,dtGOOD)

plt.figure(1)
plt.subplot(311)
plt.plot(freq,np.abs(mp), label="datos", c="green")
plt.xlabel("Frecuencias")
plt.ylabel("F(f(t))")
plt.legend()
plt.subplot(312)
plt.plot(freqGOOD,abs(F11) , label="inter cua", c="red")
plt.xlabel("Frecuencias")
plt.ylabel("F(f(t))")
plt.legend()
plt.subplot(313)
plt.plot(freqGOOD, abs(F22), label="inter cub")
plt.xlabel("Frecuencias")
plt.ylabel("F(f(t))")
plt.legend()
plt.savefig("EstupinanAndres_TF_Interpola.pdf")

#Aca ploteo los datos de signal.dat filtrados usando el paquete para que se pueda apreciar mejor la diferencia entre la transformada de la funcion completa, y las obtenidas de las interpolaciones.


#10

print("La transformada de la funcion original tiene picos con frecuencias menores al alejarnos del cero de la funcion comparado con la transformada de las interpolaciones")

#11

FBAD1=mp
FBAD2=mp


for x in range(len(freqBAD)):
	if(abs(freqBAD[x])>1000):
		FBAD1[x]=0
for x in range(len(freq)):
	if(abs(freq[x])>500):
		FBAD2[x]=0

F111=F11
F112=F11

for x in range(len(freqGOOD)):
	if(abs(freqGOOD[x])>1000):
		F111[x]=0
for x in range(len(freqGOOD)):
	if(abs(freqGOOD[x])>500):
		F112[x]=0

F221=F22
F222=F22

for x in range(len(freqGOOD)):
	if(abs(freqGOOD[x])>1000):
		F221[x]=0
for x in range(len(freqGOOD)):
	if(abs(freqGOOD[x])>500):
		F222[x]=0
for x in range(len(freq)):
	if(abs(freq[x])>1000):
		mp[x]=0

filtradaBAD1= np.fft.ifft(FBAD1)
filtradaBAD2= np.fft.ifft(FBAD2)

filtradaGOOD11=np.fft.ifft(F111)
filtradaGOOD12=np.fft.ifft(F112)

filtradaGOOD21=np.fft.ifft(F221)
filtradaGOOD22=np.fft.ifft(F222)

plt.figure()
plt.subplot(211)
plt.plot(Ax, np.real(filtradaBAD1)+np.imag(filtradaBAD1), label="Signal filtrada 1000hz Datos")
plt.plot(X, np.real(filtradaGOOD11)+np.imag(filtradaGOOD11), label="Signal filtrada 1000hz Quadratic")
plt.plot(X, np.real(filtradaGOOD21)+np.imag(filtradaGOOD21), label="Signal filtrada 1000hz Cubic")
plt.legend()

plt.subplot(212)
plt.plot(Ax, np.real(filtradaBAD2)+np.imag(filtradaBAD2), label="Signal filtrada 500hz Datos")
plt.plot(X, np.real(filtradaGOOD12)+np.imag(filtradaGOOD12), label="Signal filtrada 500hz Quadratic")
plt.plot(X, np.real(filtradaGOOD22)+np.imag(filtradaGOOD22), label="Signal filtrada 500hz Cubic")
plt.legend()
plt.savefig("EstupinanAndres_2Filtros.pdf")

#Aca ploteo los datos de signal.dat filtrados usando el paquete para que se pueda apreciar mejor la diferencia entre la transformada de la funcion completa, y las obtenidas de las interpolaciones.

































