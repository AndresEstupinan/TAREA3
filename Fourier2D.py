import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import NoNorm
from matplotlib.colors import LogNorm

# 1 Almaceno la imagen en un arreglo de numpy usando imread

arbol= plt.imread("arbol.png")
arbolf= fftpack.fft2(arbol)
arbolf2= np.fft.fftshift(arbolf)

#2 Realizo la transformada de fourier de los datos, y habia decidido no obtener frecuencias ni centrar el ruido porque pienso realizar el filtro con las amplitudes, pero tras la clase del martes, la profe nos dijo que era mejor usar este metodo. 

plt.figure()
plt.imshow(np.abs(arbolf2),cmap="gray")#, norm=NoNorm()) 
plt.colorbar()
plt.savefig("EstupinanAndres_FT2D.pdf")


#3 Despues de mucha prueba y error encuentro que el error se encuentra en estos rangos por lo que decido realizar el filtro de esta manera. Si decidiera poner unicamente mayor a 4100 se perderia informacion de la imagen, y quedaria con menos color. Esto es porque hay un pico mayor a 4100 que no es el del ruido que lleva informacion de la imagen. 


for x in range(len(arbolf[:,1])):
	for y in range(len(arbolf[1,:])):
		if(np.abs(arbolf[x,y])>4100 and np.abs(arbolf[x,y])<9100):
			arbolf[x,y]=0

arbolf2= np.fft.fftshift(arbolf)


#uso LogNorm para mostrar la transformada filtrada

plt.figure()
plt.imshow(np.abs(arbolf2), cmap="gray", norm=LogNorm(vmin=4))#, norm=NoNorm()) 
plt.colorbar()
plt.savefig("EstupinanAndres_FT2D_filtrada.pdf")

#4 Realizo la transformada inversa de fourier, y muestro la imagen resultante.

arbol2=fftpack.ifft2(arbolf).real
plt.figure()
plt.imshow(arbol2,cmap="gray")
plt.savefig("EstupinanAndres_Imagen_filtrada.pdf")			

















