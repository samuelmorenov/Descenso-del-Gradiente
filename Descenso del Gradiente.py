#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import scipy as sc

import matplotlib.pyplot as plt


# In[34]:


# Definimos la funcion sobre la que vamos a trabajar
func = lambda th: np.sin(1/2 * th[0]**2-1/4*th[1]**2+3)* np.cos(2*th[0]+1-np.e**th[1])

# Generamos la visualizacion usando dos vectores 
# con valores que van de 2 a -2
res = 100

# Vector X
_X = np.linspace(-2, 2, res)
# Vector Y
_Y = np.linspace(-2, 2, res)

# Matriz de 100*100
_Z = np.zeros((res,res))

# Añadimos los valores de la funcion en la matriz
for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[iy, ix] = func([x, y])
        
# Visualizamos la matriz 
# plt.contour(_X, _Y, _Z)
plt.contourf(_X, _Y, _Z, 100)

# Añadimos la barra de valores a la imagen
plt.colorbar()

# Creamos un punto aleatorio dentro de la matriz
Theta = np.random.rand(2) * 4 - 2

plt.plot(Theta[0], Theta[1], "o", c="white")

_T = np.copy(Theta)

h=0.001
lr = 0.01 #learning rate: distancia que te mueves en cada iteracion

gradiente = np.zeros(2)

for _ in range(10000):

    # Calculamos la derivada parcial
    for it, th in enumerate(Theta):
        _T = np.copy(Theta) 
        
        _T[it] = _T[it] + h
        derivada = (func(_T) - func(Theta)) /h
        gradiente[it] = derivada

    Theta = Theta - lr * gradiente
    
    #print(func(Theta))
    
    if(_ % 50 == 0):
        plt.plot(Theta[0], Theta[1], ".", c="red")
    

# Añadimos el punto a la imagen
plt.plot(Theta[0], Theta[1], "o", c="green")
plt.show()

