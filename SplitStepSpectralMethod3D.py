# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:08:46 2022

@author: Pedro Rueda


citar si encuentro util el paquete de transformada de Hankel en python: 
    S. G. Murray and F. J. Poulin, “hankel: A Python library for performing simple and accurate Hankel transformations”,
    Journal of Open Source Software, 4(37), 1397, https://doi.org/10.21105/joss.01397

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # import colormap tools
from scipy.fft import fft


# for pulse splitting
def field158(A,r,w0,f,k0,Chirp,t,tp):
    
    return A*np.exp(-(np.power(r/w0,2)) - 1j*k0*(np.power(r,2))/(2*f) - (1+1j*Chirp)*np.power(t/tp,2))     # initial pulse
 
    
 
def splitstep(E0):   
    
    return fft(E0)

w0 = 100e-6
lam = 800e-9
c = 299792458 #m/s
n = 1.328
kbar = 2*3.1416 * n / lam 
k1 = 1j /(2* kbar)
foco = 0.3 # metros
E0 = 1
k0 = 2*3.1416 / lam
Chirp = -1
tp = 50e-15
k0_ord2 = 241e-28                           # s2/m
k1_2 = - 1j*k0_ord2 / 2


# eje espacial
xa = 300e-6
Nx = 100
dx = (xa - (-xa)) / Nx
x = np.linspace(-xa/2, xa/2,Nx,dtype=complex)
r = np.sqrt(np.power(x,2) + np.power(x,2))
dr = dx

# eje temporal
ta = 400e-15    # segundos
Nt = 2**8
dt = (ta - (-ta)) / Nt
t = np.linspace(-ta/2, ta/2,Nt,dtype=complex)    # segundos

# eje de propagacion
z_end = 3e-2
Nz = 1500
dz = z_end / Nz

r,t = np.meshgrid(r,t)

'''
El eje espacial son las columnas y el temporal las filas.
'''

E0 = field158(E0,r,w0,foco,k0,Chirp,t,tp)


plt.figure(figsize=(9, 9),dpi=100)
plt.imshow(np.real(E0), cmap = 'RdBu_r', vmin=0, vmax=1)
plt.colorbar()
plt.xlabel("Eje Espacial")
plt.ylabel("Eje Temporal")
plt.show()

Ef = splitstep(E0)

plt.figure(figsize=(9, 9),dpi=100)
plt.imshow(np.real(Ef), cmap = 'RdBu_r', vmin=0, vmax=1)
plt.colorbar()
plt.xlabel("Eje Espacial")
plt.ylabel("Eje Temporal")
plt.show()