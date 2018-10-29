import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import NoNorm
from matplotlib.colors import LogNorm

# 1

arbol= plt.imread("arbol.png")
arbolf= fftpack.fft2(arbol)

#2

plt.figure()
plt.imshow(np.abs(arbolf),cmap="gray")#, norm=NoNorm()) 
plt.colorbar()
plt.savefig("EstupinanAndres_FT2D.pdf")


			

















