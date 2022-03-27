# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:26:11 2022

@author: Pedro Rueda, Federico Furch 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # import colormap tools
import time

# from NumericalMethods import complex_crank_nicolson2d


def campo1(x,y,w):
    # ecuacion gaussiana comun y corriente, se usó como prueba. 
    return np.exp(-(np.power(r/w,2)))

                  
# for pulse splitting
def field158(A,r,w,f,k0,Chirp,t,tp):
    
    return A * np.exp(-(np.power(r/w,2)) - 1j*k0*(np.power(r,2))/(2*f) - (1+1j*Chirp)*np.power(t/tp,2))     # initial pulse
    

def field_in_MPC(E0,r,w,Rin,k0,Chirp,T,tp):
    return E0 * np.exp(-(np.power(r/w,2))) * np.exp(-1j *k0*np.power(r,2)/(2*Rin)) * np.exp(-(1+1j*Chirp)*(T**2/tp**2)) * np.exp(-1j*w*T)  #defition of field in the time domain




def campo160(E0,tx,ty,tp,k0_2,chirp,z=0): 
    
    Zds = (tp**2 )/ (2*k0_2)
    Tz = tp * np.sqrt(((1 + chirp*(z/Zds))**2 ) + (z/Zds)**2)
    PHIz = np.arctan(((1+chirp**2)*z + chirp)/Zds)
    
    return E0 * (tp/Tz) * np.exp(-(((tx/Tz)**2)+(ty/Tz)**2)*(1 + 1j*(chirp + (1 + chirp**2)*z/Zds)) - 1j*PHIz)
    

def complex_crank_nicolson2d(f,k1,dz,zin,z,Nz,dr,k2 = 0, k3 = 0,disperssion = False,k_disp=0,dt=0,N1=0,N2=0,photons=0,Nonlinear = False):
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
            # calculo de intensidad de entrada
            # E2I = 0.5*c*eps0 #constant to go from field amplitude to intensity
            # It = E2I*np.abs(campo_in[:,int(Nx/2)])**2
            # Ir = E2I*np.abs(campo_in[int(Nt/2),:])**2
            # Eft = Ef[:,int(Nx/2)]
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
    iteraciones = Nz+2
    crossZ = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    # densidad de electrones
    rhoZ = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    #guardaré cada aumento de rho en una matriz respecto a su posicion en el espacio
    drho = np.zeros((iteraciones,Nt+2,Nr+2),dtype=complex)
    
    
    contador = 0
    while(zin < z): 
        
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
        
        zin = zin + dz
        print(f"lugar en Z = {zin}")
        contador = contador + 1
        print(f"z = {contador} de {iteraciones}")
        
    return f, crossZ,drho,rhoZ




'''
                                    Apartado para obtener algunos datos a preferencia
'''

def consultaParametros(diametroEntrada, foco, longitudOnda):
        
    
    dd = (fffoco*longOnda)/dd0          # tamaño de haz de entrada 
    print(f"\n\nDiametro de entrada = {dd} metros")
    
    # Angulo de difracción
    deltaTheta = longOnda / dd
    print(f"Angulo de difracción = {deltaTheta} radianes")
    
    # Distancia Rayleigh
    ZZr = np.pi*(dd0**2) / longOnda
    print(f"Distancia Rayleigh = {ZZr} metros\n")

    return dd, ZZr

# Tamaño de haz de entrada teniendo el tamaño en el foco
# para ejecutar en spyder5, seleccione solo estas lineas de codigo
# y presione F9 
# Mientras no lo use, puede comentar esta partecita

dd0 = 100e-6         # micrometros radio en el foco
dd0 = 2*dd0
fffoco = 30e-2    # metros
longOnda = 800e-9    # metros

dd, ZZr = consultaParametros(dd0, fffoco, longOnda)

'''
                                    Parametros del haz de entrada 

            Puede si bien, usar w = dd si conoce tan solo el diametro en el foco y halla el diametro
            en la entrada o puede también comentar esta linea y setear la entrada del haz directamente
            en la variable "w" que de igual forma más adelante se calcula el tamaño en el foco con el 
            nombre de variable dd0.
'''


w = dd/2                                 # tamaño del haz de entrada calculado, a la salida del telescopio ya colimado
# w = 1.553896e-3                      # tamaño del haz de entrada conocido, a la salida del telescopio ya colimado
lam0 = 800e-9
eps0 = 8.854e-12                        #vacuum permitivity
c = 299792458                           # Velocidad de la luz m/s
n = 1.328                               # indice de refracción lineal                                   #Indice de refraccion del vacio
# n = 1
foco = 30e-2#0.018                              # foco en metros
k0 = (2*np.pi * n) / lam0                     # Numero de Onda
Chirp = -1
FWHM = 50e-15           
tp = FWHM/(np.sqrt(2 * np.log(2)))      # pulse width assuming Gaussian shape
frecuencia_laser = 1000 #Hz
Energy = 0.29e-3                        #pulse energy at first cavity mirror in Joules



Pot_laser = Energy*frecuencia_laser     # potencia media laser

Pp = 0.94 * (Energy / tp)               # potencia pico del pulso asumiendo forma gaussiana.
I0 = (2*Energy/(np.pi*w**2)/(np.sqrt(np.pi/2)*tp))    #intensity at input mirror
E0 = np.sqrt(2*I0/(c*eps0))                         #field amplitude at input mirror



# eje espacial trasversal
xa = w                # metros
Nx = 100
dx = (xa - (-xa)) / Nx
x = np.linspace(-xa/2, xa/2,Nx,dtype=complex)
y = np.linspace(-xa/2, xa/2,Nx,dtype=complex)
r = np.sqrt(np.power(x,2) + np.power(y,2))
dr = np.abs(np.real(r[0]) - np.real(r[1]))

# eje temporal
ta = 400e-15    # Ventana temporal en segundos
Nt = 2**8       # puntos de muestra en el eje temporal
dt = (ta - (-ta)) / Nt
t = np.linspace(-ta/2, ta/2,Nt,dtype=complex)    # segundos

# eje de propagacion
z_end = 3e-2                # metros
Nz = 1000                   #pasos de propagacion Z
# ejeZ = np.linspace(-z_end, z_end,Nz)
ejeZ = np.linspace(0e-2,z_end,Nz)
dz = np.abs(ejeZ[0] - ejeZ[1])               #propagation step


#Rayleigh del haz colimado y enfocado entrando en la MPC                    
dd0 = (foco*lam0)/w                     # tamaño del haz en el foco
Zr0 = np.pi * (dd0**2) / lam0           # Rayleigh en el foco en metros 
Rin0 = -foco*(1+(Zr0/(-foco))**2)       # Radio de curvatura de mi haz gaussiano




# condición para que Zr0 funcione, el diametro del haz en el foco debe complir con la siguiente condicion
# es decir no puede ser demasiado pequeño

print(f"Tamaño del haz en el foco teniendo en cuenta solo la distancia focal luego de colimación = {dd0} metros")
print(f"El foco no es demasiado pequeño, verdad? = {dd0 >= ((2*lam0) / np.pi)}")
print(f"Radio de curvatura de mi haz entrante = {Rin0} radianes")
print(f"Cauntas veces se reduce el tamaño del haz en el foco respecto a la entrada? = {np.round(w/dd0)} veces")

# Terminos lineales
k0_ord2 = 241e-28                       # s2/m
k1_2 = - 1j*k0_ord2 / 2 
k1 = 1j /(2* k0)


# Terminos no lineales
n2 = 1.6e-20                # antes (e-16)indice no lineal de refraccion (cm2 / W) -> m2 / W
Bk = 8e-64                  # antes (e-50)  cm7 / W4 -> m7 / W4 
photons = 5
N1 = 1j*(dd0*n2) / c
N2 = Bk / 2

# Pin = np.pi*(dd0**2)*I
Pcr = (3.77*np.pi*n) / (2*n2*k0**2)     #Potencia critica para self-focusing en haz gaussiano
Pcr2 = (n*c*lam0**2)/(8*np.pi*n2)
# Zc = 0.367*Zr0 / (np.sqrt(np.pow(np.sqrt(Pin/Pcr) - 0.852,2) - 0.0219))
print(f"Potencia critica para self-focusing = {Pcr}")
print(f"Potencia critica = {Pcr2}")

print("\n\n\n######## Datos de mi pulso de entrada ########")
print(f"\nEl haz conservará su diametro trasversal hasta \nRayleigh del haz de entrada = {(w**2) / lam0} metros")
print(f"Longitud de onda = {lam0} metros")
print(f"n0 = {n}")
print(f"\nEnergia del pulso = {Energy} Joules")
print(f"Potencia media del laser = {Pot_laser} W")
print(f"FWHM = {FWHM} fs")
print(f"Ancho del pulso gaussiano = {tp} fs")
print(f"Potencia pico del laser = {Pp} W")
print(f"Intensidad del haz de entrada = {I0} W/m2")
print(f"Amplitud del haz de entrada = {E0}")
print(f"Ventana temporal = {ta} segundos")
print(f"Ventana espacial = {xa} metros")
print(f"Tamaño trasversal del haz de entrada = {w} [m]")
print(f"Distancia de propagacion por round trip = {z_end} metros")
print(f"dz = {dz}")
print(f"Distancia de propagacion total = {z_end}")
print(f"Pasos en R = {Nx}; Pasos en T = {Nt}; Pasos en Z = {Nz}")
print(f"Chirp = {Chirp}")
'''
                        Confocal Cavity characteristics, mode definitions
'''
# ref = 0.97                  #mirror reflectivity (value between 0.0 and 1.0 - assumed flat in freq. for silver ~0.96-0.97
# round_trips = 0             #number of round trips in the cavity
# extra_pass = True           #True for adding an extra through the focus (half round trip)
# f1 = 0.5*300e-3             #300e-3    #focal distance mirror 1 (all distances in m)
# f2 = 0.5*300e-3             #300e-3   #focal distance mirror 2
# z_end = 0.999*2*(f1+f2)     #mirror separation en metros
# L = z_end                   # longitud de cavidad, espejo a espejo en metros
# g1 = 1-L/(2*f1)             #following definitions according to Siegman's book
# g2 = 1-L/(2*f2)

# if g1 == g2:                # Confocal symmetric case
#     wo = np.sqrt((L*lam0/np.pi)*np.sqrt((1+g1)/(4*(1-g1))))  #cavity waist
#     z0 = L/2                                                  #position of waist

# else:
#     wo = np.sqrt(lam0*L/np.pi*np.sqrt(g1*g2*(1-g1*g2)/(g1+g2-2*g1*g2)**2))     #cavity waist
#     z0 = g2*(1-g1)/(g1+g2-2*g1*g2)*L                                            #position of waist

# Zr = (np.pi*(wo**2))/lam0         #Rayleigh length en el modo de la cavidad
# z1 = z0                           #distancia desde el foco al espejo 1.

# win =  wo*np.sqrt(1+(z1/Zr)**2)   #beam size at 1st mirror
# Rin = z1*(1+(Zr/z1)**2)           #radius of curvature at 1st mirror


# print("\n\n\n######## Confocal MPC Caracteristics ########")
# print("\nLa siguiente información son las caracteristicas de la MPC, setee \nsu haz de entrada TEM00 con diametro de entrada especificado")
# print(f"\nSetee el diametro del haz de entrada en = {win} [m]")
# print(f"\n0 <= g1g2 <= 1 : Estabilidad {0 <= g1*g2 <= 1}") # condicion de estabilidad de la cavidad
# print(f"Rayleight de la cavidad = {Zr} [m]")
# print(f"Distancia desde foco al espejo 1 = {z1} [m]")
# print(f"Radio de curvatura del haz en espejo 1 = {Rin}")
# print(f"Tamaño de haz en foco = {wo} [m]")
# print(f"foco1 = {f1} metros \nfoco2 = {f2} metros")
# print(f"Numero de round trips = {round_trips}")
# print(f"Distancia de propagacion total = {(round_trips+1)*z_end}")


#hasta este momento, obtendria el campo referente al laser 
#al momento de tocar el primer espejo de la cavidad.

dis = True
nonl = False 
gaussiano_espacial = False

if dis:
    
    R,T = np.meshgrid(r,t)
    # campo_in = field_in_MPC(E0,R,w,Rin0,k0,Chirp,T,tp)        # campo entrante a la MPC
    # campo_in = field158(E0,R,w,foco,k0,Chirp,T,tp)
    
    
    zin = ejeZ[0]
    Zr0 = 0.5*k0*dd0**2
    Wf = dd0*foco / np.sqrt((foco**2) + (Zr0**2))
    Zf = (k0*Wf**2) / 2
    Wz = Wf*np.sqrt(1 + ((zin - ejeZ[int(Nz/2)])/Zf)**2)
    Rz = zin-ejeZ[int(Nz/2)] + (Zf**2)/(zin - ejeZ[int(Nz/2)])
    phi = 1j*np.arctan((zin-ejeZ[int(Nz/2)])/Zf)
    
    
    
    campo_in = E0*np.exp(-np.power(R/Wz,2))*np.exp(-(1+1j*Chirp)*np.power(T/tp,2))#*np.exp(-phi)*np.exp(1j*(k0*(np.power(R,2)/(2*Rz))))*np.exp(-1j*k0*zin)*(dd0/Wz)
    
    titulo1 = "Perfil de intensidad temporal del pulso"
    titulo2 = "Perfil de intensidad espacial del pulso"
    EjeX = "Espacio [micrometros]"
    EjeY = "Tiempo [Femtosegundos]"
    Nx = Nx
    Ny = Nt
    x = x
    y = t
    
    extent = np.real(np.min(x))*1e5, np.real(np.max(x))*1e5, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.real(np.abs(campo_in)), cmap = cm.inferno,extent=extent)
    plt.colorbar()
    plt.title("Pulso entrada en R vs T")
    plt.xlabel(EjeX)
    plt.ylabel(EjeY)
    # plt.text(55, 160, f"w0 = {w0}[m]")#" \nlam = {lam0}[m] \nChirp = {Chirp}\ntp = {tp}[fs]")
    plt.show()
    
elif gaussiano_espacial:
    X,Y = np.meshgrid(x,y)
    # Wz = dd0*np.sqrt(1 + (ejeZ[0] / Zr0)**2)
    # Rz = ejeZ[0]*(1+(Zr0/ejeZ[0])**2)
    R = np.sqrt(np.power(X,2) + np.power(Y,2))
    # phi = 1j*np.arctan(ejeZ[0]/Zr0)
    #------------------------------------------------------

    zin = ejeZ[0]
    Zr0 = 0.5*k0*dd0**2
    Wf = dd0*foco / np.sqrt((foco**2) + (Zr0**2))
    Zf = (k0*Wf**2) / 2
    Wz = Wf*np.sqrt(1 + ((zin - ejeZ[int(Nz/2)])/Zf)**2)
    Rz = zin-ejeZ[int(Nz/2)] + (Zf**2)/(zin - ejeZ[int(Nz/2)])
    phi = 1j*np.arctan((zin-ejeZ[int(Nz/2)])/Zf)
    
    
    campo_in = E0*(dd0/Wz)*np.exp(-np.power(R/Wz,2))*np.exp(1j*(k0*(np.power(R,2)/(2*Rz))))*np.exp(-1j*k0*zin)*np.exp(-phi)# haz gaussiano comun

    titulo1 = "Perfil de intensidad espacial del pulso Y"
    titulo2 = "Perfil de intensidad espacial del pulso X"
    EjeX = "Espacio [micrometros]"
    EjeY = "Espacio [micrometros]"
    Nx = Nx
    Ny = Nx
    x = x
    y = y
    
    extent = np.real(np.min(x))*1e5, np.real(np.max(x))*1e5, np.real(np.min(x))*1e5, np.real(np.max(x))*1e5
    plt.figure(figsize=(9, 9),dpi=100, frameon=True)
    plt.imshow(np.abs(campo_in), cmap = cm.inferno,extent=extent)
    plt.colorbar()
    plt.title("Pulso entrada en x vs x")
    plt.xlabel(EjeX)
    plt.ylabel(EjeY)
    plt.show()
    
# Ef = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(campo_in))) #to frequency domain
 

'''
                                    Graficas
'''

# grafica del perfil de intensidad temporal del pulso
plt.figure()
fig, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,2]},constrained_layout=True)
axes[1].set_title(titulo1)
axes[1].set_xlabel(EjeY)
axes[1].set_ylabel("Intensidad")
axes[1].plot(np.real(y),np.real(np.abs(campo_in[:,int(Nx/2)])))

# # grafica del perfil de intensidad espacial del pulso
axes[0].set_title(titulo2)
axes[0].set_xlabel(EjeX)
axes[0].set_ylabel("Intensidad")
axes[0].plot(np.real(x),np.real(np.abs(campo_in[int(Ny/2),:])))
plt.show()


'''
                                    Propagación en MPC
'''

campo_in[:,0] = 0 + 0j
campo_in[0,:] = 0 + 0j
campo_in[Nt-1,:] = 0 + 0j
campo_in[:,Nx-1] = 0 + 0j

print(dr)

EE, ZZ, drho, rhoZ = complex_crank_nicolson2d(campo_in,k1,dz,zin,z_end,Nz,dr,disperssion = dis,k_disp=k1_2,dt=dt,N1=N1,N2=N2,photons=photons,Nonlinear=nonl)



# extent = np.real(np.min(x))*1e6, np.real(np.max(x))*1e6, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
# plt.figure(figsize=(9, 9),dpi=100)
# plt.imshow(np.abs(EE), cmap = cm.inferno,extent=extent)#,vmax = 4e7, vmin = -1e7)
# plt.colorbar()
# plt.xlabel(EjeX)
# plt.ylabel(EjeY)
# plt.show()



# # grafica del perfil de intensidad temporal del pulso
# plt.figure()
# fig, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,2]},constrained_layout=True)
# axes[1].set_title(titulo1)
# axes[1].set_xlabel(EjeY)
# axes[1].set_ylabel("Intensidad")
# axes[1].plot(np.real(y),np.real(np.abs(EE[:,int(Nx/2)])))

# # # grafica del perfil de intensidad espacial del pulso
# axes[0].set_title(titulo2)
# axes[0].set_xlabel(EjeX)
# axes[0].set_ylabel("Intensidad")
# axes[0].plot(np.real(x),np.real(np.abs(EE[int(Ny/2),:])))
# plt.show()



# # sacar franja en columna central y graficarla respecto a z
# solucion2 = []
# for i in range(int(Nz)):

#     solucion2.append(ZZ[i][:,int(Nx/2)])

# extent = ejeZ[0]*1e5, np.real(np.max(ejeZ))*1e5, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
# plt.figure(figsize=(5, 1),dpi=100)
# plt.imshow(np.abs(np.transpose(solucion2[:])), cmap = cm.inferno)#,vmax = 5e10, vmin = 0, extent = extent)
# plt.colorbar()
# plt.xlabel(" Espacio micras ")
# plt.ylabel(" Tiempo fs ")
# plt.show()

















# intensidad = []

# for i in range(len(ZZ)):
#     intensidad.append(np.real(np.amax(ZZ[i])))
    

# plt.figure()
# plt.plot(intensidad)
# plt.show()










# for i in range(len(ZZ)+2):
    
#     plt.figure()
#     plt.plot(np.real(x),np.real(np.abs(ZZ[i][int(Nx/2),:])))
#     plt.xlabel(EjeX)
#     plt.ylabel("Intensidad")
#     plt.show()
#     time.sleep(0.0001)


# for i in range(len(ZZ)+2):
    
#     plt.figure()
#     plt.plot(np.real(t),np.real(np.abs(ZZ[i*5][:,int(Nx/2)])))
#     plt.xlabel(EjeY)
#     plt.ylabel("Intensidad")
#     plt.show()
#     time.sleep(0.0001)


'''


# drho_2 = drho
# rhoZ_2 = rhoZ




plt.figure()
fig, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,2]},constrained_layout=True)
# axes[0].set_title("Perfil de intensidad temporal del pulso")
# axes[0].set_xlabel("Tiempo")
# axes[0].set_ylabel("Intensidad")
# axes[0].plot(t,np.real(It))

# grafica del perfil de intensidad espacial del pulso
axes[1].set_title("Perfil de intensidad espacial del pulso")
axes[1].set_xlabel("Espacio")
axes[1].set_ylabel("Intensidad")
axes[1].plot(x,np.abs(EE[int(Nx/2),:]))
plt.show()




camposs = ZZ
'''
# # extent2 = np.min(x)*1e5, np.max(x)*1e5, np.min(y)*1e5, np.max(y)*1e5
# for i in range(len(ZZ)+2):
    
#     plt.figure(figsize=(9, 9),dpi=100)
#     plt.imshow(np.abs(ZZ[i]), cmap = cm.inferno)#,vmax = 2e31, vmin = 0)
#     plt.colorbar()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
#     time.sleep(0.0001)


# sacar franja en fila central y graficarla respecto a z
# solucion = []
# for i in range(Nz):

#     solucion.append(ZZ[i][int(Nt/2),:])


# plt.figure(figsize=(5, 1),dpi=500)
# plt.imshow(np.real(np.transpose(solucion[500:1200])), cmap = 'RdBu_r', vmin=0, vmax=3e30)#cm.inferno)
# plt.colorbar()
# plt.xlabel(" Espacio micras ")
# plt.ylabel(" Tiempo fs ")
# plt.show()
    



 




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