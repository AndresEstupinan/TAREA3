import numpy as np
import matplotlib.pyplot as plt


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

