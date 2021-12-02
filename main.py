# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import crank_nicolson as CN
import matplotlib.pyplot as plt
# Ejercicio de prueba "metodos numericos para ingenieria 7th edicion crank-nicolson"
x = [0,2,4,6,8,10]
dx = 2
Nx = len(x) - 2 # defino los nodos internos del alambre
k = 0.835           # cm2 / seg
dt = 0.1           # tiempo en segundos
t = 0.2
f = [100,0,0,0,0,50]

# Aolicando el metodo de Crank-Nicolson
sol = CN.crank_nicolson1d(f, k,dt,t, dx, Nx)


# el resultado me está indicando la temperatura final de la barra cuando ah pasado t tiempo. 
print(sol)




# Preparacion valores para campo E que representa un haz gaussiano

# pi = 3.14159265359                  # numero pi
# n0 = 1.33                           # indice de refraccion del agua pura
# f0 = 0.5*300e-3                     # distancia de foco en metros
# lam0 = 820e-9                       # longitud de onda central en metros
# w0 = 2*lam0                         # ancho del haz en el foco
# k0 = n0*2*pi / lam0                 # numero de onda central
                    
# xmax = 1.5e-3 
# Nx = 100
# x = np.linspace(-xmax,xmax,Nx)      # arreglo espacial
# dx = x[1]-x[0]                      # paso espacial en grilla                      
# z = 10                              # metros
# dz = 0.1
# delta = dz / (4*k0*(dx**2))
# E0 = 1

# E = E0*np.exp(-(x / w0)**2 - 1j*((k0*x**2) / (2*f0)))


# # solucion por Crank-Nicolson
# k = 1j / (2*k0)
# sol = CN.crank_nicolson1d(E, k, dz, z, dx, Nx-2)
