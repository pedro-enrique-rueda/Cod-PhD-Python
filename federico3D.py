#simple 3D propagation model: Split-step algorithm. Takes into account diffraction and dispersion to all orders. Nonlinear propagation based solely on Kerr nonlinearity (SPM and self-focusing).

#Created by Federico Furch (FF) on 04.12.2020
#Last modification: FF on 06.12.2020

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import colormap_fede as colf #colormap edited by F.F. Can be replaced by any built-in colormap
import time

################################################################################
#function definitions
################################################################################

# Fourier transform from time to frequency domain
def time2freq(ft,**params):
    if 'filter' in params:
        filter = params['filter']
        F = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ft*filter)))
    else:
        F = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ft)))
    return F

# Fourier transform from frequency to time domain    
def freq2time(ff,**params):
    if 'filter' in params:
        filter = params['filter']
        F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(ff*filter)))
    else:
        F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(ff)))
    return F

# Definition of supergaussian of center xo, width wx, order n (n must be even and positive)
def supergaussian(x,xo,wx,n):
    return np.exp(-(x-xo)**n/wx**n)

# Dispersio properties of different materials
def material_properties(f,material):
        lam = np.linspace(0.3,1.5,500)      #wavelength (um);lam is the internal lambda axis
        f1 = 100e12 #f1 and f2 define the frequencies (in Hz between which the dispersion functions will be calculated
        f2 = 700e12
        if1 = (np.abs(f-f1)).argmin()   #find index corresponding to f1 in f
        if2 = (np.abs(f-f2)).argmin()   #find index corresponding to f2 in f
        nu = f[if1:if2] #frequency vector running from f1 to f2
        
        # Load material file

        if material == 'argon':
            A1=6.432135e-5
            A2=2.8606021e-2
            A3=144
            n1 = 1+A1+A2/(A3-1/lam**2) #index of refraction on internal wavelength axis
            n2 = 9.7e-24   #nonlinear index (m^2/W) Wahlstrand et. al., PRL 109, 113904 (2012)

            
            
        elif material == 'helium':
            A1=6.432135e-5
            A2=2.8606021e-2
            A3=144
            n1 = 1+A1+A2/(A3-1/lam**2) #index of refraction on internal wavelength axis
            n2 = 3.1e-25   #nonlinear index (m^2/W) Wahlstrand et. al., PRL 109, 113904 (2012)

        
        elif material == 'Fused-Silica':
            B1=0.69616630
            B2=0.40794260
            B3=0.89747940
            C1=0.0684043**2
            C2=0.116241**2
            C3=9.896161**2
            n1 = np.sqrt(1+B1*lam**2/(lam**2-C1)+B2*lam**2/(lam**2-C2)+B3*lam**2/(lam**2-C3)) #index of refraction on internal wavelength axis
            n2 = 3e-20   #nonlinear index (m^2/W)

            
        else:
            error('error: method not recognized')
            
        c = 3e8 #speed of light in m/s
        w = 2*np.pi*nu;        #positive angular frequency axis (rad/s)
        #index and group delay as a function of angular frequency
        dw = w[1]-w[0] 
        n = np.interp(2*np.pi*c/w,1e-6*lam,n1) #interpolation of n to w axis
        dndw = np.zeros((len(nu),),dtype='float') #empty array for derivative of n with respect to w
        dndw[0] = (n[1]-n[0])/dw
        dndw[len(nu)-1] = (n[len(n)-1]-n[len(n)-2])/dw
        for ii in range(1,len(n)-1):
            dndw[ii] = 0.5/dw*(n[ii+1]-n[ii-1]) #filling the derivative array
        vg1=1/(n/c+w/c*(dndw)) #use n(w) and dn/dw(w) to compute group velocity array
        
        #the next few steps place n(w) and vg(w) into f axis (all values outside [f1,f2] are set to 1)
        n1 = np.ones(np.shape(f),dtype='float')
        n1[if1:if2] = n
        vg = c*np.ones(np.shape(f),dtype='float')
        vg[if1:if2] = vg1
        
        return n1,n2,vg  


###################################################################################################
# Definitions and initialization
###################################################################################################

# Constants in SI units
eps0 = 8.854e-12 #vacuum permitivity
c = 3e8 #speed of light
E2I = 0.5*c*eps0 #constant to go from field amplitude to intensity


# Laser input parameters
lam0 = 780e-9 #center wavelengths
Energy = 0.29e-3 #pulse energy at first cavity mirror
FWHM = 40e-15 #initial pulse duration Full Width at Half Maximum in seconds
wt = FWHM/(np.sqrt(2 * np.log(2))) #pulse width assuming Gaussian shape
GDD = -100e-30 #in sec^2 #input group delay dispersion
TOD = 15000e-45; #in sec^3 #input third order dispersion

    
# For grids definitions
Nt = 1200 #points in time grid
Nx = 64 #points in spatial grid (Nx**2)
Ns = 50 #number of steps
tmax = 0.35e-12
xmax = 2e-3


# Cavity characteristics, mode definitions
ref = 0.97 #mirror reflectivity (value between 0.0 and 1.0 - assumed flat in freq. for silver ~0.96-0.97
round_trips = 5 #number of round trips in the cavity
extra_pass = True #True for adding an extra through the focus (half round trip)
f1 = 0.5*300e-3 #300e-3    #focal distance mirror 1 (all distances in m)
f2 = 0.5*300e-3 #300e-3   #focal distance mirror 2
L = 0.975*2*(f1+f2) #mirror separation
dL = L/Ns   #propagation step
g1 = 1-L/(2*f1) #following definitions according to Siegman's book
g2 = 1-L/(2*f2) 
if g1 == g2:
    wxo = np.sqrt(L*lam0/np.pi*np.sqrt((1+g1)/(4*(1-g1)))) #cavity waist
    z1 = L/2 #position of waist
else:
    wxo = np.sqrt(lam0*L/np.pi*np.sqrt(g1*g2*(1-g1*g2)/(g1+g2-2*g1*g2)**2)) #cavity waist
    z1 = g2*(1-g1)/(g1+g2-2*g1*g2)*L            #position of waist
Zr = np.pi*wxo**2/lam0            #Rayleigh length
z = z1          #distance from telescope to first mirror (to define input radius)
win =  wxo*np.sqrt(1+(z/Zr)**2)    #beam size at 1st mirror
Rin = z*(1+(Zr/z)**2)     #radius of curvature at 1st mirror


# Definition of grids
t = np.linspace(-tmax,tmax,Nt) #time array
dt = t[1]-t[0]
fmax = 1/dt
x = np.linspace(-xmax,xmax,Nx) #spatial array
dx = x[1]-x[0]
fxmax = 1/dx 
[X,Y,T] = np.meshgrid(x,x,t) #3D space-time grids
f = np.fft.fftshift(np.fft.fftfreq(Nt))*fmax #frequency fector
f[np.abs(f).argmin()] = f[np.abs(f).argmin()+1]/2 #remove f=0 to avoid dividing by 0
fx = np.fft.fftshift(np.fft.fftfreq(Nx))*fxmax #spatial-frequency vector
[Fx,Fy,Ft] = np.meshgrid(fx,fx,f) #3D space-time-frequency grids
frac_fil = 0.90 #to define width of supergaussian
gaussian_order = 20
filter_3D = supergaussian(X,0,frac_fil*xmax,gaussian_order) * supergaussian(Y,0,frac_fil*xmax,gaussian_order) * supergaussian(T,0,frac_fil*tmax,gaussian_order) #supergaussian for 3D grids


# Arrays for propagation
dz = dL #propagation step
wo = 2*np.pi*c/lam0 #central angular frequency
fo = c/lam0 #central frequency
[n,n2,vg] = material_properties(f,'argon') #call material dispersion properties
io = (np.abs(f-fo)).argmin() #find location of central frequency in frequency vector
Kw = 2*np.pi*Ft / c * n #definition of k(w)
fun_phase = (2*np.pi/Kw)**2 * (Fx**2 + Fy**2) #function to define propagation phase
fun_phase[fun_phase>=1]=1 #avoid imaginary values (for frequencues where spectrum = 0)
om = 2*np.pi*Ft-wo  #(angular frequency array centered at 0 ->(w-w0)
phase_prop = (Kw * np.sqrt(1 -fun_phase) - om/vg[io] )* dz #linear propagation phase
ko = wo / c #central wavenumber
gamma = ko * n2 #constant for nonlinear propagation


# Input pulses (at cavity mirror 1. beam size and wavefront determine by cavity mode)
phase_init = 1/2 * GDD * om**2 + 1/6 * TOD * om**3 #input  chirp 
wx = win #set beam width to cavity mode
Io = 2*Energy/(np.pi*wx**2)/(np.sqrt(np.pi/2)*wt) #intensity at input mirror
Eo = np.sqrt(2*Io/(c*eps0)) #field amplitude at input mirror
Et = Eo * np.exp(-(X**2+Y**2)/wx**2) * np.exp(-T**2/wt**2) * np.exp(-1j*wo*T) * np.exp(1j *ko*(X**2+Y**2)/(2*Rin)) #defition of field in the time domain
Ef = time2freq(Et) #to frequency domain
Ef = Ef * np.exp(1j * phase_init) #add initial chirp
Et = freq2time(Ef) #back to time domain


# Phases for reflection in cavity mirrors 
phase_mirror1 = - ko * (X**2+Y**2) / (2*f1)
phase_mirror2 = - ko * (X**2+Y**2) / (2*f2)


################################################################################
#Run propagation code
################################################################################

#loop for round trips
for jj in range(round_trips):

    tic = time.time() # timing of computation
    
    Bint = 0 #B-integral
    #reflection in mirror 1
    Et = np.sqrt(ref) * Et * np.exp(1j * phase_mirror1)

    #free space propagation
    for ii in range(Ns):
        #dispersion / difraction
        Ef = time2freq(Et,filter=filter_3D) #note that I applied filters now to avoid reflection at edge of grids
        Ef = Ef * np.exp(1j*phase_prop)
        Et = freq2time(Ef,filter=filter_3D)
        
        #nonlinear step
        It = E2I*np.absolute(Et)**2
        Et = Et * np.exp(1j * gamma * It *dz)
        Bint = Bint + ko * n2 * It.max() * dz
    
    #reflection in mirror 2
    Et = np.sqrt(ref) * Et * np.exp(1j * phase_mirror2)
    
    #free space propagation
    for ii in range(Ns):
        #dispersion / difraction
        Ef = time2freq(Et,filter=filter_3D)
        Ef = Ef * np.exp(1j*phase_prop)
        Et = freq2time(Ef,filter=filter_3D)
        
        #nonlinear step
        It = E2I*np.absolute(Et)**2
        Et = Et * np.exp(1j * gamma * It *dz)
        Bint = Bint + ko * n2 * It.max() * dz 
    
    toc = time.time() #close timing computation

    #print info
    lapse = (toc-tic)/60
    text = 'time per iteration: %1.1f' %lapse
    text0 = 'iteration %2.0f' %(jj+1)
    text1 = 'B-integral per pass %2.2f' %(Bint)
    print(text0)
    print(text)
    print(text1)

#add extra pass through focal plane?
if extra_pass:
    #reflection in mirror 1
    Et = ref * Et * np.exp(1j * phase_mirror1)

    #free space propagation
    for ii in range(Ns):
        #dispersion / difraction
        Ef = time2freq(Et,filter=filter_3D)
        Ef = Ef * np.exp(1j*phase_prop)
        Et = freq2time(Ef,filter=filter_3D)
        
        #nonlinear step
        It = E2I*np.absolute(Et)**2
        Et = Et * np.exp(1j * gamma * It *dz)


Ef = time2freq(Et,filter=filter_3D) #final conversion to frequency domain


################################################################################
#analyze results
################################################################################

Sf =  np.sum(np.sum(np.abs(Ef)**2,0),0) #spectrum in frequency vector
pulse = np.sum(np.sum(np.abs(Et)**2,0),0) #pulse intensity profile
beam = np.sum(np.abs(Et)**2,axis=2) #output beam
beam_f = np.sum(np.abs(Ef)**2,axis=2) #output beam in frequency domain

#plot pulse
plt.figure()
plt.plot(t * 1e15,pulse,'b')
plt.xlabel('t (fs)')
plt.ylabel('Intensity')
plt.show()

#plot beam
cm = colf.ff_color()
plt.figure()
plt.pcolor(1e3*x,1e3*x,beam,cmap=cm)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.show()

#plot beam in frequency domain
plt.figure()
plt.pcolor(fx,fx,beam_f,cmap=cm)
plt.xlabel('kx (1/m)')
plt.ylabel('ky (1/m)')
plt.show()

# Calculate spectrum in wavelength axis
Sf_plus = np.zeros((np.int(Nt/2),),dtype = 'float') #array for positive frequencies
f_plus = np.zeros((np.int(Nt/2),),dtype = 'float')  #array for positive frequencies
Sf_plus = Sf[np.int(Nt/2):] #Spectrum for positive frequencies
f_plus = f[np.int(Nt/2):] #positive frequencies
lam = np.linspace(450e-9,1050e-9,2000) #definition of wavelength array
SL_fun = interpolate.interp1d(c/f_plus,Sf_plus) #interpolation freq to lambda
SL = SL_fun(lam) #Spectrum as a function of lambda
lam = lam*1e9 #m to nm
SL = SL/SL.max() #normalization

#plot spectrum
plt.figure()
plt.plot(lam,SL,'r')
plt.xlabel('$\lambda$ (nm)')
plt.ylabel('Spectral power density')
plt.show()

# Calculate spatio-spectral distribution
Exyf  = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(Ef * filter_3D,axes=0),axis=0),axes=0) #E(x,y,f)
Sxyf = np.abs(Exyf)**2 #spatio-spectral in x,y,f
Sxyf_plus = np.zeros((Nx,Nx,np.int(Nt/2)),dtype = 'float') #for definining positive frequency E(f)
Sxyf_plus = Sxyf[:,:,np.int(Nt/2):] #S(x,y,f) at positive frequencies
SxyL_fun = interpolate.interp1d(1e9*c/f_plus,Sxyf_plus,axis=2) #interpolate to lambda axis
SxyL = SxyL_fun(lam) #spatio-spectral in wavelength array
SxL = np.sum(SxyL[:,np.abs(x<50e-6)],axis=1) #spectrum over line at beam center 
SxL = SxL/SxL.max() #normalization

#plot spatio-spectral distribution
plt.figure()
plt.pcolor(lam,1e3*x,SxL,cmap=cm)
plt.ylabel('x (mm)')
plt.xlabel('$\lambda$ (nm)')
plt.show()

#plot beam profile
px = np.sum(beam,axis=0)
px = px/px.max()
plt.figure()
plt.plot(1e3*x,px)
plt.xlabel('x (mm)')
plt.ylabel('Normalized intensity')
plt.show()
