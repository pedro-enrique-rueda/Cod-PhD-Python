
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:23:13 2022

@author: Pedro Rueda / Federico Furch

                        Caso de prueba:
                            ecuacion 102, e implementacion 
                            seccion 3.1.2 Crank-Nicolson
                            Practioner's guide...
                            
                            Beam Splitting para verificar la participacion 
                            correcta de la seccion no-lineal 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # import colormap tools
import time


def consultaParametros(diametroFoco, foco, longitudOnda):
        
    
    diametroEntrada = (fffoco*longOnda)/diametroFoco          # diametro de haz de entrada 
    print(f"\n\nDiametro de entrada = {diametroEntrada*1e2} cm")
    print(f"d/d0 = {diametroEntrada/diametroFoco} veces")
    
    # Angulo de difracción
    deltaTheta = longOnda / diametroEntrada
    print(f"Angulo de difracción de haz colimado = {deltaTheta*1e3} mili-radianes")
    
    # Distancia Rayleigh de haz colimado
    Zrcol = (diametroEntrada**2) / longitudOnda
    print(f"Rayleigh del haz colimado = {Zrcol*1e2} m")
    

    return diametroEntrada, Zrcol





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
            eps0 = 8.854e-12                        #vacuum permitivity
            c = 299792458                           # Velocidad de la luz m/s
            n = 1.328                               # indice de refracción lineal del agua
            It = 0.5*eps0*c*n*np.absolute(crossZ[contador])**2
            It_1 = 0.5*eps0*c*n*np.absolute(crossZ[contador-1])**2
            
            # densidad de electrones
            rho_nt = 0.54e25                            # 1/m3
            cross_sec_kphotons = 2.81e-128              # m16 / (W8 * s)  esto para 8 photones ejemplo con oxigeno
            Tc = 350e-15                                # segundos
            gamma = 5.6e-24                             # m2  la ecu es : (k0*w0*Tc)/(n0*rho_c*(1 + (w0**2)*(Tc**2))
            rho_c = 1.7e-27                             # critical plasma density ecuacion: eps0*me*((2*pi*c) / (c*lam0)**2)
            Ui = 12                                     # eV
            
            drho[contador] = cross_sec_kphotons*(( (It**photons) + (It_1**photons))/2)*(rho_nt - rhoZ[contador-1]) + (gamma/Ui)*rhoZ[contador-1]*(It + It_1)/2
            rhoZ[contador] = rhoZ[contador-1] + drho[contador]
            
            
            Nn = N1*It*crossZ[contador] - N2*(np.power(It,(photons-1)))*crossZ[contador] + (-gamma/2)*rhoZ[contador]*crossZ[contador]
            Nn_1 = N1*It_1*crossZ[contador-1] - N2*(np.power(It_1,(photons-1)))*crossZ[contador-1] + (-gamma/2)*rhoZ[contador-1]*crossZ[contador-1]

            N_total = A*Nn - B*Nn_1

            return N_total,drho,rhoZ, N1,N2

    
        else:
            
            re = np.zeros((Nt+2,Nr+2),dtype=complex)
            return re,drho,rhoZ, N1,N2                     
        
    
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
        
        xx1,drho,rhoZ,N1,N2 = nolinealidad(N1,N2,crossZ,contador,Nonlinear,photons,rhoZ,drho)
        
        nolin = dt*dz*xx1
        
        # First Half of the method
        for j in range(1,Nr+1): # columns
            k = 0
            for i in range(1,Nt+1): # rows

                if(k==0):
                    b[0] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam*f[i,j+1] + lam*f[i,j-1] + lam2*f[i-1,j]) 
                    k = k + 1
                    continue

                if(k==Nt-1):
                    b[Nr-1] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam2*f[i+1,j] + lam*f[i,j-1] + lam*f[i,j+1]) 
                    k = k + 1
                    continue
                else:
                    b[k] = 2*(1 + omega*(-lam + k2 + k3))*f[i,j] + omega*(lam*f[i,j+1] + lam*f[i,j-1]) 
                    k = k + 1
                    continue
            
            
            res = np.dot(L_Invert1,b)
            
            for i in range(1,Nt+1):
                f[i,j] = np.around(res[i-1],4)
                
        f[1:Nt+1,:] = f[1:Nt+1,:] + np.dot(L_Invert1,nolin[1:Nt+1,:])
 
        
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
        
    return f, crossZ,drho,rhoZ,N1,N2,nolin



radioFoco = 100e-6      # en metros
dd0 = 2*radioFoco       # micrometros diametro en el foco
fffoco = 2e-2          # metros
longOnda = 800e-9       # metros

TamañoHazEntrada, DistanciaRayleigh = consultaParametros(dd0, fffoco, longOnda)




w = TamañoHazEntrada/2                 # radio del haz de entrada calculado, a la salida del telescopio ya colimado
# w = (1.553896e-3)/2                      # tamaño del haz de entrada conocido, a la salida del telescopio ya colimado
lam0 = 800e-9
eps0 = 8.854e-12                        #vacuum permitivity
c = 299792458                           # Velocidad de la luz m/s
n = 1.328                               # indice de refracción lineal del agua
foco = 2e-2                            # foco en metros
k0 = (2*np.pi * n) / lam0               # Numero de Onda
Chirp = -10
FWHM = 50e-15                           # fs           
tp = FWHM/(np.sqrt(2 * np.log(2)))      # pulse width assuming Gaussian shape
frecuencia_laser = 1000                 # Hz
Energy = 290e-3                          # pulse energy at first cavity mirror in Joules
Pp = 0.94 * (Energy / tp)               # potencia pico del pulso asumiendo forma gaussiana.
Pot_laser = Energy*frecuencia_laser     # potencia media laser Joules*Hz Tambien puede expresarse np.pi*(w0**2)*I0/2


# eje espacial trasversal
xa = 15*w                # metros
Nx = 100
dx = (xa - (-xa)) / Nx
x = np.linspace(-xa/2, xa/2,Nx,dtype=complex)
y = np.linspace(-xa/2, xa/2,Nx,dtype=complex)
r = np.sqrt(np.power(x,2) + np.power(y,2))
dr = np.abs(np.real(r[0]) - np.real(r[1]))
X,Y = np.meshgrid(x,y)

# eje temporal
ta = 400e-15    # Ventana temporal en segundos
Nt = 2**8       # puntos de muestra en el eje temporal
dt = (ta - (-ta)) / Nt
t = np.linspace(-ta/2, ta/2,Nt,dtype=complex)    # segundos
R,T = np.meshgrid(r,t)


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

# eje de propagacion
zin = 0e-2
z_end = 3e-2                # metros
Nz = 1000                   #pasos de propagacion Z
ejeZ = np.linspace(zin,z_end,Nz)
dz = np.abs(ejeZ[0] - ejeZ[1])               #propagation step

# para jugar con la potencia critica y la potencia de entrada
Pcr = 3.77*np.pi*n / (2*n2*k0**2)       # potencia critica para self focusing
Pot_laser = 1*Pcr



dis = 1
nonl = 0 
gaussiano_espacial = 0

if dis:
    
    w0 = radioFoco
    Zr0 = np.pi*(w0**2)*n/lam0                                                          # Rayleigh tambien puede expresarse 0.5*k0*w0**2
    # I0 = 2*Pot_laser / (np.pi * (w0)**2)                                                # W / m2 Espacial, no temporal
    # Ir = I0*np.power(w0/w,2)*np.exp(-2*np.power(R,2)/(w**2))                            # Intensidad del haz en la entrada W / m2.
    # It = 0.5*eps0*c*n*np.exp(-1.385*np.power(T/tp,2))
    # E0 = np.sqrt(2*I0 / (c*eps0))                                                       # Amplitud del campo electrico en el foco.
    
    I0 = 2*Energy/(np.pi*w0**2)/(np.sqrt(np.pi/2)*tp)                           #intensity at input mirror
    E0 = np.sqrt(2*I0/(c*eps0))                                                 #field amplitude at input mirror
    RadCur = -foco*(1 + np.power(Zr0 / (-foco),2)) 
        
    # campo_in = E0*(dd0/Wz)*np.exp(-np.power(R/Wz,2))*np.exp(-phi)*np.exp(1j*(k0*(np.power(R,2)/(2*Rz))))*np.exp(-1j*k0*zin)*np.exp(-(1+1j*Chirp)*np.power(T/tp,2))
    # Ecuacion 158 para pulse splitting, Comentar esta linea cuando no se esté usando
    Ertz = E0*np.exp(-np.power(R/(w0),2)) *np.exp(-(1+1j*Chirp)*np.power(T/tp,2))* np.exp(- 1j*k0*np.power(R,2)/(2*foco))
          
    titulo1 = "Perfil de intensidad temporal del pulso"
    titulo2 = "Perfil de intensidad espacial del pulso"
    EjeX = "Espacio [micrometros]"
    EjeY = "Tiempo [Femtosegundos]"
    Nx = Nx
    Ny = Nt
    x = x
    y = t
    
    Pin = np.pi*(w0**2)*I0/2  
    Zc = 0.367*Zr0 / np.sqrt(np.power((np.sqrt(Pin/Pcr) - 0.852),2) - 0.0219)
    print(Zc*1e2)
    print(f"\nPotencia critica para self-focusing: Pcr = {Pcr:.1E} W")
    print(f"Potencia a la entrada del foco: Pin = {Pin:.1E} W")
    print(f"Distancia focal no lineal Zc = {Zc*1e2} cm ¿desde el foco?")
    
    print(f"\nTamaño haz de entrada: w = {2*w*1e2} cm")
    print(f"Tamaño haz en foco: w0 = {2*w0*1e2} cm")
    
    # print(f"\nIntensidad a la entrada del haz gaussiano espacial: Ir = {np.real(np.amax(Ir))*1e4:.1E} W/cm2")
    print(f"Intensidad en el foco espacial del haz: I0 = {I0*1e4:.1E} W/cm2")
    
    print(f"\nAmplitud del campo en el foco: E0 = {E0:.1E}")
    print(f"Rayleigh en el foco: Zr0 = {Zr0*1e2} cm")
    
    extent = np.real(np.min(x))*1e5, np.real(np.max(x))*1e5, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.abs(Ertz), cmap = cm.inferno,extent=extent,vmax = np.abs(np.real(np.amax(Ertz))) , vmin = np.abs(np.real(np.amin(Ertz))))
    plt.colorbar()
    plt.title(f"Perfil del haz inicial a {-foco} m")
    plt.xlabel(EjeX)
    plt.ylabel(EjeY)
    plt.show()

    # LLamar al metodo Crank-Nicolson
    campo_in = Ertz
    
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
    
    
      
    EE, ZZ, drho, rhoZ,N1,N2,nolin = complex_crank_nicolson2d(campo_in,k1,dz,zin,z_end,Nz,dr,disperssion=True,k_disp=k1_2,dt=dt,N1=N1,N2=N2,photons=photons,Nonlinear=bool(nonl))
    
    nolineal = nolin
    
    plt.figure()
    plt.plot(nolineal[:,int(Nx/2)])
    plt.show()
    
    extent22 = np.real(np.min(x))*1e6, np.real(np.max(x))*1e6, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.abs(nolineal), cmap = cm.inferno,extent=extent22,vmax = np.abs(np.real(np.amax(nolineal))) , vmin = np.abs(np.real(np.amin(nolineal))))
    plt.colorbar()
    plt.xlabel("Eje X")
    plt.ylabel("Eje T")
    plt.show()


    
    extent2 = np.real(np.min(x))*1e6, np.real(np.max(x))*1e6, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.abs(EE), cmap = cm.inferno,extent=extent2,vmax = np.abs(np.real(np.amax(EE))) , vmin = np.abs(np.real(np.amin(EE))))
    plt.colorbar()
    plt.xlabel("EjeX [micras]")
    plt.ylabel("EjeY [micras]")
    plt.show()
    
    # grafica del perfil de intensidad temporal del pulso
    plt.figure()
    fig, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,2]},constrained_layout=True)
    axes[1].set_title(titulo1)
    axes[1].set_xlabel(EjeY)
    axes[1].set_ylabel("Intensidad")
    axes[1].plot(np.real(y),np.real(np.abs(EE[:,int(Nx/2)])))
    
    # # grafica del perfil de intensidad espacial del pulso
    axes[0].set_title(titulo2)
    axes[0].set_xlabel(EjeX)
    axes[0].set_ylabel("Intensidad")
    axes[0].plot(np.real(x),np.real(np.abs(EE[int(Ny/2),:])))
    plt.show()
    
    # sacar franja en columna central y graficarla respecto a z
    solucion2 = []
    for i in range(int(Nz/2)):
    
        solucion2.append(ZZ[i*2][:,int(Nx/2)])
    
    extent = ejeZ[0]*1e3, np.real(np.max(ejeZ))*1e3, np.real(np.min(T))*1e14, np.real(np.max(T))*1e14
    plt.figure(figsize=(5, 1),dpi=100)
    plt.imshow(np.abs(np.transpose(solucion2[:int(1*Nz)])), cmap = cm.inferno, extent=extent,vmax = 5e9, vmin = 0)
    plt.colorbar()
    plt.xlabel(" Espacio mm ")
    plt.ylabel(" Tiempo fs ")
    plt.show()

    # extent4 =  np.real(np.min(x))*1e6, np.real(np.max(x))*1e6, np.real(np.min(T))*1e15, np.real(np.max(T))*1e15
    # for i in range(len(ZZ)+2):
        
    #     plt.figure(figsize=(9, 9),dpi=100)
    #     plt.imshow(np.abs(ZZ[i]), cmap = cm.inferno, extent = extent4,vmax = 2e8, vmin = 0)
    #     plt.colorbar()
    #     plt.title(f"{i}")
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()
    #     time.sleep(0.0001)
        
elif gaussiano_espacial:
    # definimos nuestro haz gaussiano Espacial, con w0 en el origen de coordenadas
    w0 = radioFoco
    Zr0 = np.pi*(w0**2)*n/lam0                                                          # Rayleigh tambien puede expresarse 0.5*k0*w0**2
    I0 = 2*Pot_laser / (np.pi * (w0)**2)                                                # W / m2 Espacial, no temporal
    Iz = I0*np.power(w0/w,2)*np.exp(-2*(np.power(X,2) + np.power(Y,2))/(w**2))          # Intensidad del haz en la entrada W / m2.
    E0 = np.sqrt(2*I0 / (c*eps0))                                                       # Amplitud del campo electrico en el foco.
    RadCur = -foco*(1 + np.power(Zr0 / (-foco),2)) 
    Erz = E0*(w0/w)*np.exp(-(np.power(X,2) + np.power(Y,2)) / (w**2))*np.exp(-1j*(k0*(-foco)))*np.exp(-1j*k0*(np.power(X,2) + np.power(Y,2))/(2*RadCur))*np.exp(-1j*np.arctan(-foco/Zr0))
     

    titulo1 = "Perfil de intensidad espacial del pulso Y"
    titulo2 = "Perfil de intensidad espacial del pulso X"
    EjeX = "Espacio [micrometros]"
    EjeY = "Espacio [micrometros]"
    Nx = Nx
    Ny = Nx
    x = x
    y = y
    
    Pin = np.pi*(w0**2)*I0/2  
    Zc = 0.367*Zr0 / np.sqrt(np.power((np.sqrt(Pin/Pcr) - 0.852),2) - 0.0219)
    
    print(f"\nPotencia critica para self-focusing: Pcr = {Pcr:.1E} W")
    print(f"Potencia a la entrada del foco: Pin = {Pin:.1E} W")
    print(f"Distancia focal no lineal Zc = {Zc*1e2} cm ¿desde el foco?")
    
    print(f"\nTamaño haz de entrada: w = {2*w*1e2} cm")
    print(f"Tamaño haz en foco: w0 = {2*w0*1e2} cm")
    
    print(f"\nIntensidad a la entrada del haz gaussiano espacial: Iz = {np.real(np.amax(Iz))*1e4:.1E} W/cm2")
    print(f"Intensidad en el foco espacial del haz: I0 = {I0*1e4:.1E} W/cm2")
    
    print(f"\nAmplitud del campo en el foco: E0 = {E0:.1E}")
    print(f"Rayleigh en el foco: Zr0 = {Zr0*1e2} cm")
    
    extent = np.real(np.min(x))*1e3, np.real(np.max(x))*1e3, np.real(np.min(y))*1e3, np.real(np.max(y))*1e3
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.abs(Erz), cmap = cm.inferno,extent=extent,vmax = np.abs(np.real(np.amax(Erz))) , vmin = np.abs(np.real(np.amin(Erz))))
    plt.colorbar()
    plt.title(f"Perfil del haz inicial a {-foco} m")
    plt.xlabel(EjeX)
    plt.ylabel(EjeY)
    plt.show()
    
    # LLamar al metodo Crank-Nicolson
    campo_in = Erz
    
    
    EE, ZZ, drho, rhoZ = complex_crank_nicolson2d(campo_in,k1,dz,zin,z_end,Nz,dx,disperssion=False,k_disp=k1_2,dt=dt,N1=N1,N2=N2,photons=photons,Nonlinear=bool(nonl))

    extent2 = np.real(np.min(x))*1e3, np.real(np.max(x))*1e3, np.real(np.min(y))*1e3, np.real(np.max(y))*1e3
    plt.figure(figsize=(9, 9),dpi=100)
    plt.imshow(np.abs(EE), cmap = cm.inferno,extent=extent2,vmax = np.abs(np.real(np.amax(EE))) , vmin = np.abs(np.real(np.amin(EE))))
    plt.colorbar()
    plt.xlabel("EjeX [micras]")
    plt.ylabel("EjeY [micras]")
    plt.show()
    
    # sacar franja en columna central y graficarla respecto a z
    solucion2 = []
    for i in range(int(Nz)):
    
        solucion2.append(ZZ[i][:,int(Nx/2)])
    
    plt.plot(x,Erz[:,int(Nx/2)],'o')
    plt.plot(x,solucion2[0],'b')
    plt.show()
    
    
    extent3 = ejeZ[0]*1e6, np.real(np.max(ejeZ))*1e6, np.real(np.min(y))*1e7, np.real(np.max(y))*1e7
    plt.figure(figsize=(5, 1),dpi=100)
    plt.imshow(np.abs(np.transpose(solucion2[:])), extent = extent3, cmap = cm.inferno,vmax = np.abs(np.real(np.amax(EE))) , vmin = np.abs(np.real(np.amin(EE))))
    plt.colorbar()
    plt.xlabel(" Espacio micras ")
    plt.ylabel(" Tiempo fs ")
    plt.show()
    
    
    extent4 = np.real(np.min(x))*1e3, np.real(np.max(x))*1e3, np.real(np.min(y))*1e3, np.real(np.max(y))*1e3
    for i in range(len(ZZ)+2):
        
        plt.figure(figsize=(9, 9),dpi=100)
        plt.imshow(np.abs(ZZ[i]), cmap = cm.inferno, extent = extent4,vmax = 2e8, vmin = 0)
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        time.sleep(0.0001)
