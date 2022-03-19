# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:26:11 2022

@author: Pedro Rueda, Federico Furch 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # import colormap tools
# from NumericalMethods import complex_crank_nicolson2d


def campo1(x,y,w0):
    # ecuacion gaussiana comun y corriente, se usó como prueba. 
    return np.exp(- ((x**2) + (y**2)) / (w0**2))

# for pulse splitting
def field158(A,r,w0,f,k0,Chirp,t,tp):
    
    return A*np.exp(-(np.power(r/w0,2)) - 1j*k0*(np.power(r,2))/(2*f) - (1+1j*Chirp)*np.power(t/tp,2))     # initial pulse
    

def field_in_MPC(E0,r,w0,Chirp,T,tp,k0,Rin):
    return E0 * np.exp(-(np.power(r/w0,2)) * np.exp((1+1j*Chirp)*(-T**2/tp**2)) * np.exp(-1j*w0*T)) * np.exp(1j *k0*(np.power(r,2))/(2*Rin)) #defition of field in the time domain




def campo160(E0,tx,ty,tp,k0_2,chirp,z=0): 
    
    Zds = (tp**2 )/ (2*k0_2)
    Tz = tp * np.sqrt(((1 + chirp*(z/Zds))**2 ) + (z/Zds)**2)
    PHIz = np.arctan(((1+chirp**2)*z + chirp)/Zds)
    
    return E0 * (tp/Tz) * np.exp(-(((tx/Tz)**2)+(ty/Tz)**2)*(1 + 1j*(chirp + (1 + chirp**2)*z/Zds)) - 1j*PHIz)
    

def complex_crank_nicolson2d(f,k1,dz,z,dr,k2 = 0, k3 = 0,disperssion = False,k_disp=0,dt=0,N1=0,N2=0,photons=0,Nonlinear = False):
    '''
        This code accept 6 arguments: function in 2D; k argument, it means any part of the equation
        apart from funciton and derivatives df/dz= k*(d2f/dx2) for example,
        dz, z, dr, Nr. 

        crank_nicolson2d(f,k,dz,t,dr,Nr)
    '''
    
    
    # funcion de no linealidad
    def nolinealidad(N1,N2,crossZ,contador,Nonlinear,photons,rhoZ,drho):
        
        if(Nonlinear and contador != 0):
                           
            A = 3/2
            B = 1/2
            # eps0 = 8.854e-12                           #vacuum permitivity
            # c = 299792458 
            # E2I = 0.5*c*eps0
            It = np.abs(crossZ[contador])
            It_1 = np.abs(crossZ[contador-1])
            
            # densidad de electrones
            rho_nt = 0.54e25                            # 1/m3
            cross_sec_kphotons = 2.81e-128              # m16 / (W8 * s)  esto para 8 photones ejemplo con oxigeno
            Tc = 350e-15                                # segundos
            gamma = 5.6e-24                             # m2  la ecu es : (k0*w0*Tc)/(n0*rho_c*(1 + (w0**2)*(Tc**2))
            rho_c = 1.7e-27                             # critical plasma density ecuacion: eps0*me*((2*pi*c) / (c*lam0)**2)
            Ui = 12                                     # eV
            
            
            drho[contador] = cross_sec_kphotons*(( (It**photons) + (It_1**photons))/2)*(rho_nt - rhoZ[contador-1]) + (gamma/Ui)*rhoZ[contador-1]*(It + It_1)/2
            rhoZ[contador] = rhoZ[contador-1] + drho[contador]
            
            
            
            Nn = N1*(It**2)*crossZ[contador] - N2*(It**(2*photons-2))*crossZ[contador] + (-gamma/2)*rhoZ[contador]*crossZ[contador]
            Nn_1 = N1*(It_1**2)*crossZ[contador-1] - N2*(It_1**(2*photons-2))*crossZ[contador-1] + (-gamma/2)*rhoZ[contador-1]*crossZ[contador-1]

            N_total = A*Nn - B*Nn_1

            return N_total,drho,rhoZ
    
        else:
            
            re = np.zeros((Nt+2,Nr+2),dtype=complex)
            return re,drho,rhoZ                     
        
    
    Nt = f.shape[0]-2
    Nr = f.shape[1]-2
    
    if disperssion:    
        lam = (k1) / (dr**2)
        lam2 = (k_disp) / (dt**2)
        omega = dz
        
        # Creacion de Matriz A
        matriz1 = 2*(1-omega*(-lam2 + k2 + k3))*np.identity(Nt,dtype=complex) - omega*lam2*np.eye(Nt,k=-1,dtype=complex) - omega*lam2*np.eye(Nt,k=1,dtype=complex)
        matriz2 = 2*(1-omega*(-lam + k2 + k3))*np.identity(Nr,dtype=complex) - omega*lam*np.eye(Nr,k=-1,dtype=complex) - omega*lam*np.eye(Nr,k=1,dtype=complex)
        
    else:
        lam = (k1) / (dr**2)
        lam2 = (k1) / (dr**2)
        omega = dz

        # Creacion de Matriz A
        matriz1 = 2*(1-omega*(-lam + k2 + k3))*np.identity(Nt,dtype=complex) - omega*lam*np.eye(Nt,k=-1,dtype=complex) - omega*lam*np.eye(Nt,k=1,dtype=complex)
        matriz2 = 2*(1-omega*(-lam + k2 + k3))*np.identity(Nr,dtype=complex) - omega*lam*np.eye(Nr,k=-1,dtype=complex) - omega*lam*np.eye(Nr,k=1,dtype=complex)
    
    
    L_Invert1 = np.linalg.inv(matriz1)
    L_Invert2 = np.linalg.inv(matriz2)
    
    # creamos el arreglo 3D para guardar cada plano en cada posición Z, lo llamaremos crossZ    
    iteraciones = int(z/dz)+2
    crossZ = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    # densidad de electrones
    rhoZ = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    #guardaré cada aumento de rho en una matriz respecto a su posicion en el espacio
    drho = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    
    contador = 0
    while(contador < iteraciones): 
        
        # vamos a guardar cada f(dz) en una matriz 3D
        crossZ[contador] = f
        
        # Creacion de Matriz b
        b = np.zeros([Nt,1],dtype=complex)
        
        xx1,drho,rhoZ = nolinealidad(N1,N2,crossZ,contador,Nonlinear,photons,rhoZ,drho)
        
        nolin = dt*dz*xx1
        
        
        # First Half of the method
        for j in range(1,Nr+1): # columns
            k = 0
            for i in range(1,Nt+1): # rows

                if(k==0):
                    b[0] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam*f[i,j+1] + lam*f[i,j-1] + lam2*f[i-1,j]) + nolin[i,j]
                    k = k + 1
                    continue

                if(k==Nt-1):
                    b[Nr-1] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam2*f[i+1,j] + lam*f[i,j-1] + lam*f[i,j+1]) + nolin[i,j]
                    k = k + 1
                    continue
                else:
                    b[k] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam*f[i,j+1] + lam*f[i,j-1]) + nolin[i,j]
                    k = k + 1
                    continue
                    
            res = np.dot(L_Invert1,b)
            
            for i in range(1,Nt+1):
                f[i,j] = np.around(res[i-1],4)
          
        b1 = np.zeros([Nr,1],dtype=complex)        
        
        # Second Half of the method
        for i in range(1,Nt+1): # rows
            k = 0
         
            for j in range(1,Nr+1): # columns
                if(k==0):
                    b1[0] = 2*(1 + omega*(-lam2 + k2 + k3))*f[i,j] + omega*(lam2*f[i+1,j] + lam2*f[i-1,j] + lam*f[i,j-1])
                    k = k + 1
                    continue

                if(k==Nr-1):
                    b1[Nr-1] = 2*(1 + omega*(-lam2 + k2 + k3))*f[i,j] + omega*(lam*f[i,j+1] + lam2*f[i-1,j] + lam2*f[i+1,j])
                    k = k + 1
                    continue
                else:
                    b1[k] = 2*(1 + omega*(-lam2 + k2 + k3))*f[i,j] + omega*(lam2*f[i+1,j] + lam2*f[i-1,j])
                    k = k + 1
                    continue
                 
            res1 = np.dot(L_Invert2,b1) #+ b2             
               
            for j in range(1,Nr+1):
                f[i,j] = res1[j-1]
        
        
        contador = contador + 1
        print(f"z = {contador} de {iteraciones}")
        
    return f, crossZ,drho,rhoZ








'''
                                    Parametros del haz de entrada 
'''

# w0 = 100e-6                     # tamaño del spot espacial del haz en el foco
w0 = 5e-3                       # tamaño del haz de entrada, a la salida del telescopio ya colimado
lam0 = 800e-9
eps0 = 8.854e-12                #vacuum permitivity
c = 299792458 #m/s
n = 1.328                       # indice de refracción lineal
kbar = 2*3.1416 * n / lam0 
k1 = 1j /(2* kbar)
k0 = 2*3.1416 / lam0
Chirp = -0
FWHM = 50e-15           
frecuencia_laser = 1000 #Hz
Energy = 0.29e-3                        #pulse energy at first cavity mirror in Joules
Pot_laser = Energy*frecuencia_laser     # potencia media laser
tp = FWHM/(np.sqrt(2 * np.log(2)))      #pulse width assuming Gaussian shape
Pp = 0.94 * (Energy / tp)               # potencia pico del pulso asumiendo forma gaussiana.
k0_ord2 = 241e-28                           # s2/m
k1_2 = - 1j*k0_ord2 / 2

#Rayleigh del haz colimado entrando en la MPC
Zr0 = (w0**2) / lam0 # en metros 

I0 = 2*Energy/(np.pi*w0**2)/(np.sqrt(np.pi/2)*tp)   #intensity at input mirror
E0 = np.sqrt(2*I0/(c*eps0))                         #field amplitude at input mirror


# eje espacial
xa = w0*3                # metros
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
ref = 0.97                  #mirror reflectivity (value between 0.0 and 1.0 - assumed flat in freq. for silver ~0.96-0.97
round_trips = 0             #number of round trips in the cavity
extra_pass = True           #True for adding an extra through the focus (half round trip)
f1 = 0.5*300e-3             #300e-3    #focal distance mirror 1 (all distances in m)
f2 = 0.5*300e-3             #300e-3   #focal distance mirror 2
z_end = 0.999*2*(f1+f2)     #mirror separation en metros
Nz = 1500                   #pasos de propagacion Z
dz = z_end/Nz               #propagation step

print("Haz de entrada")
print(f"El haz conservará su diametro trasversal hasta \nRayleigh del haz de entrada = {Zr0} metros")
print(f"\nEnergia del pulso = {Energy} Joules")
print(f"Potencia media del laser = {Pot_laser} W")
print(f"FWHM = {FWHM} fs")
print(f"Ancho del pulso gaussiano = {tp} fs")
print(f"Potencia pico del laser = {Pp} W")
print(f"Intensidad del haz de entrada = {I0} W/m2")
print(f"Amplitud del haz de entrada = {E0}")
print(f"Ventana temporal = {ta} segundos")
print(f"Ventana espacial = {xa} metros")
print(f"Distancia de propagacion por round trip = {z_end} metros")
print(f"dz = {dz}")
print(f"foco1 = {f1} metros \nfoco2 = {f2} metros")
print(f"Numero de round trips = {round_trips}")
print(f"Distancia de propagacion total = {(round_trips+1)*z_end}")
print(f"Pasos en R = {Nx}; Pasos en T = {Nt}; Pasos en Z = {Nz}")
print(f"Chirp = {Chirp}")


'''
                        Confocal Cavity characteristics, mode definitions
'''
L = z_end                   # longitud de cavidad, espejo a espejo en metros
g1 = 1-L/(2*f1)             #following definitions according to Siegman's book
g2 = 1-L/(2*f2)

 
if g1 == g2:                # Confocal symmetric case
    wxo = np.sqrt((L*lam0/np.pi)*np.sqrt((1+g1)/(4*(1-g1))))  #cavity waist
    z0 = L/2                                                  #position of waist

else:
    wxo = np.sqrt(lam0*L/np.pi*np.sqrt(g1*g2*(1-g1*g2)/(g1+g2-2*g1*g2)**2))     #cavity waist
    z0 = g2*(1-g1)/(g1+g2-2*g1*g2)*L                                            #position of waist

Zr = np.pi*wxo**2/lam0             #Rayleigh length en el modo de la cavidad
z1 = z0                            #distancia desde el foco al espejo 1.

win =  wxo*np.sqrt(1+(z1/Zr)**2)    #beam size at 1st mirror
Rin = z1*(1+(Zr/z1)**2)              #radius of curvature at 1st mirror

print(f"0 <= g1g2 <= 1 : {0 <= g1*g2 <= 1}") # condicion de estabilidad de la cavidad
print(f"\nRayleight de la cavidad = {Zr} metros")
print(f"Distancia desde foco al espejo 1 = {z1} m")
print(f"Beam size at 1st mirror = {win} m")
print(f"Radio de curvatura en espejo 1 = {Rin}")
print(f"Tamaño de haz en foco = {wxo}")

R,T = np.meshgrid(r,t)


# hasta este momento, obtendria el campo referente al laser 
#al momento de tocar el primer espejo de la cavidad.

# campo_in = field_in_MPC(E0,R,w0,Chirp,T,tp,k0,Rin)        # campo entrante a la MPC
 


'''
                                    Graficas
'''


# calculo de intensidad de entrada
E2I = 0.5*c*eps0 #constant to go from field amplitude to intensity
It = np.abs(campo_in[:,int(Nx/2)])**2
Ir = np.abs(campo_in[int(Nt/2),:])**2



# grafica del perfil de intensidad temporal del pulso
plt.figure()
fig, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,2]},constrained_layout=True)
axes[0].set_title("Perfil de intensidad temporal del pulso")
axes[0].plot(np.real(t),np.real(It))

# grafica del perfil de intensidad espacial del pulso
axes[1].set_title("Perfil de intensidad espacial del pulso")
axes[1].plot(np.real(x),np.real(Ir))
plt.show()


# grafica de imagen en 2D de perfil de intensidad espacial vs temporal
extent = np.min(x)*1e5, np.max(x)*1e5, np.min(t)*1e15, np.max(t)*1e15

plt.figure(figsize=(9, 9),dpi=100, frameon=False)
plt.imshow(np.real(campo_in), cmap = cm.inferno,extent=extent) #  vmin=-1, vmax=1,
plt.colorbar()
plt.title("Pulso entrada en r vs t")
plt.xlabel("Espacio [micrometros]")
plt.ylabel("Tiempo [femtosegundos]")
plt.text(55, 160, f"w0 = {w0}[m] \nlam = {lam0}[m] \nChirp = {Chirp}\ntp = {tp}[fs]")
plt.show()


'''

# Terminos no lineales
n2 = 1.6e-20                                                   # antes (e-16)indice no lineal de refraccion (cm2 / W) -> m2 / W
Bk = 8e-64                                                    # antes (e-50)  cm7 / W4 -> m7 / W4 
photons = 5
N1 = 1j*(w0*n2) / c
N2 = Bk / 2


EE, ZZ, drho, rhoZ = complex_crank_nicolson2d(campo_in,k1,dz,z_end,dr,disperssion = True,k_disp=k1_2,dt=dt,N1=N1,N2=N2,photons=photons,Nonlinear=True)

drho_2 = drho
rhoZ_2 = rhoZ

plt.figure(figsize=(9, 9),dpi=100)
plt.imshow(np.real(EE), cmap = 'RdBu_r')#,vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

camposs = ZZ

import time

for i in range(len(ZZ)+2):
    
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.real(ZZ[i]), cmap = cm.inferno)#, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    time.sleep(0.0001)


# sacar franja en fila central y graficarla respecto a z
solucion = []
for i in range(Nz):

    solucion.append(ZZ[i][int(Nt/2),:])


plt.figure(figsize=(5, 1),dpi=500)
plt.imshow(np.real(np.transpose(solucion[500:1200])), cmap = 'RdBu_r', vmin=0, vmax=1.5)#cm.inferno)
plt.colorbar()
plt.xlabel(" Espacio micras ")
plt.ylabel(" Tiempo fs ")
plt.show()
    


# sacar franja en columna central y graficarla respecto a z
solucion2 = []
for i in range(Nz):

    solucion2.append(ZZ[i][:,int(Nx/2)])


# plt.figure(figsize=(5, 1),dpi=500)
# plt.imshow(np.real(np.transpose(solucion2[500:])), cmap = reds) #cm.inferno)#, vmin=-1, vmax=1)
# plt.colorbar()
# plt.xlabel(" Espacio micras ")
# plt.ylabel(" Tiempo fs ")
# plt.show()
 
'''



''' 
En el siguiente bloque, retratamos el fenomeno de difracción de una función gaussiana 2D
'''

# w0 = 4e-6
# xa = 20e-6
# dx = 0.2e-6
# points_on_x = int((xa - (-xa)) / dx)
# x,y = np.meshgrid(np.linspace(-xa/2, xa/2,points_on_x,dtype=complex),np.linspace(-xa/2, xa/2,points_on_x,dtype=complex))
# c = 299792458 

# lam = 1e-6
# n = 1.455
# # dz = 0.5e-6
# dz = 1e-6
# z_end = 10e-6  
# kbar = 2*3.1416 * n / lam 
# k = 1j /(2* kbar)
# f = 0.3 # metros
# E0 = 1

# # campo1 de prueba 
# v_in = campo1(x,y,w0)

# plt.figure(figsize=(9, 9),dpi=100)
# plt.imshow(abs(v_in), cmap = cm.coolwarm, vmin=0, vmax=1)
# plt.colorbar()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()



# EE, ZZ = complex_crank_nicolson2d(v_in,k,dz,z_end,dx)

# plt.figure(figsize=(9, 9),dpi=100)
# plt.imshow(abs(EE), cmap = cm.coolwarm, vmin=0, vmax=1)
# plt.colorbar()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# import time

# for i in range(len(ZZ)+2):
    
#     plt.figure(figsize=(9, 9),dpi=100)
#     plt.imshow(abs(ZZ[i]), cmap = cm.coolwarm, vmin=0, vmax=1)
#     plt.colorbar()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
#     time.sleep(0.1)


# # 
''' 
En el siguiente bloque, retratamos el fenomeno de disperción de 2do orden
de una función gaussiana 2D
'''


# w = 100e-6    #metros
# ta = 200e-15    # segundos
# dt = 0.2e-15    #segundos

# # teniendo el dt
# points_on_t = int((ta - (-ta)) / dt)    # segundos

# #teniendo la cantidad de puntos
# Nt = 100
# dt = (ta - (-ta)) / Nt
# t = np.linspace(-ta/2, ta/2,Nt)    # segundos
# c = 299792458   # m/s 
# tx,ty = np.meshgrid(np.linspace(-ta/2, ta/2,Nt,dtype=complex),np.linspace(-ta/2, ta/2,Nt,dtype=complex))

# lam = 800e-9
# n = 1.328
# # dz = 0.3e-6
# z_end = 200*2e-2    # metros
# Nz = 100
# dz = z_end / Nz  
# kbar = 2*3.1416 * n / lam 
# k = 1j /(2* kbar)
# f = 0.3 # metros
# E0 = 1

# tp = 50e-15
# chirp = -8
# k0_ord2 = 241e-30                           # fs2
# k2 = - 1j*k0_ord2 / 2
# v_in = campo160(E0,tx,ty,tp,k0_ord2,chirp) 
  
# plt.figure(figsize=(9, 9),dpi=100)
# plt.imshow(abs(v_in), cmap = cm.coolwarm, vmin=0, vmax=1)
# plt.colorbar()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# # terminos no lineales
# n2 = 1.6e-16                 # indice no lineal de refraccion (cm2 / W)
# Bk = 8e-50                   # cm7 / W4 
# photons = 5
# N1 = 1j*(w*n2) / c
# N2 = Bk / 2
# Nonlinear = False


# E, sol2 = complex_crank_nicolson2d(v_in,k2,dz,z_end,dt)

# plt.figure(figsize=(9, 9),dpi=100)
# plt.imshow(abs(E), cmap = cm.coolwarm, vmin=0, vmax=1)
# plt.colorbar()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()  
    
# import time

# for i in range(len(sol2)):
    
#     plt.figure(figsize=(9, 9),dpi=100)
#     plt.imshow(abs(sol2[i]), cmap = cm.coolwarm, vmin=0, vmax=1)
#     plt.colorbar()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
#     time.sleep(0.1)