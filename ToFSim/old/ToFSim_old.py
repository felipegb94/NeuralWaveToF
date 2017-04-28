# Python imports
import sys
import math
# Library imports
import numpy as np
# import pandas as pd
# from scipy import signal
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')


# Parameters
c = 299792458. # Speed of light
N = 10000 # Number of points to represent modulation and demodulation functions
pi = np.pi

# Mod and demod parameters
shift = 0.
f = 10000000. # 10Mhz
T = 1/f # period
n_periods = 1000 # number of integration periods
integration_time = n_periods*T # n periods
omega = 2*pi * f # angular frequency
max_depth = c * T / 2
x = np.linspace(0, integration_time, N)

# Scene parameters (for now this is for a single pixel so only one depth)
depth = max_depth/5 # 5 meters
beta = 1. # reflectance of pixel
L_a = 1. # Ambient illumination 
shift = 2*depth/c # phase shift

print "For frequency: {} hz, Max depth: {} meters".format(f,max_depth)

# Create modulation and demodulation functions
M = (np.cos(omega*x) + 1)*0.5
D1 = (np.cos(omega*x + 0) + 1)*0.5 # Demodulation function 1  
D2 = (np.cos(omega*x + (2*pi/3)) + 1)*0.5  # Demodulation function 2
D3 = (np.cos(omega*x + (4*pi/3)) + 1)*0.5  # Demodulation function 3


# Shifted function
M_shifted = (np.cos(omega*(x - shift)) + 1)*0.5
# Received radiance on sensor 
L = (beta * M_shifted) + L_a  
# Calculate the brightness measurements
B1 = L.dot(D1)  
B2 = L.dot(D2)
B3 = L.dot(D3)
print "Brightness measurements over {} periods: ({},{},{})".format(n_periods,B1,B2,B3)

# Photon noise for each brightness measurement
stddev_photon1 = np.sqrt(B1)
stddev_photon2 = np.sqrt(B2)
stddev_photon3 = np.sqrt(B3)
noise_photon1 = np.random.normal(0,stddev_photon1)
noise_photon2 = np.random.normal(0,stddev_photon2)
noise_photon3 = np.random.normal(0,stddev_photon3)

# Todo: Add Read Noise + Quantization


B = np.array([B1+noise_photon1,B2+noise_photon2,B3+noise_photon3])

# ADD Gain and Sensor Saturation

# Read and Analog-Digital-Conversion Noise depends on the gain and sensor saturation (scene indep)
# noise_read1 = np.random.normal(0,stddev_photon1)
# noise_read2 = np.random.normal(0,stddev_photon2)
# noise_read3 = np.random.normal(0,stddev_photon3)

# C Matrix of book chapter
C = np.matrix(	[[1,np.cos(0),np.sin(0)],
				[1,np.cos(2*pi/3),np.sin(2*pi/3)],
				[1,np.cos(4*pi/3),np.sin(4*pi/3)]
				])
print B
print C
X = np.linalg.solve(C, B)
print X

phi = np.arccos(X[1]/(np.sqrt((X[1]*X[1]) + (X[2]*X[2]))))
print "phi = {}".format(phi)
print "real depth = {}".format(depth)
print "recovered depth = {}".format(phi*c/(2*omega))


# fig, ax = plt.subplots(nrows=1,ncols=1)
# ax.plot(x,mod_f, label='Mod',linewidth=2.0)
# ax.plot(x,I_shifted, label='Demod',linewidth=2.0)
# ax.legend(shadow=True,fontsize=14)
# ax.set_ylim([0,1.2*np.max([np.max(mod_f),np.max(I_shifted)])])
# ax.set_title('Modulation/Demodulation',fontsize=14,fontweight='bold')
# ax.set_xlabel('time (s)',fontsize=14,fontweight='bold')
# ax.set_ylabel('magnitude',fontsize=14,fontweight='bold')
# plt.show()

# fig, ax = plt.subplots(nrows=1,ncols=1)
# ax.plot(x,R1, label='Demod1',linewidth=2.0)
# ax.plot(x,R2, label='Demod2',linewidth=2.0)
# ax.plot(x,R3, label='Demod3',linewidth=2.0)
# ax.legend(shadow=True,fontsize=14)
# ax.set_ylim([0,1.2*np.max([np.max(R1),np.max(R2)])])
# ax.set_title('Modulation/Demodulation',fontsize=14,fontweight='bold')
# ax.set_xlabel('time (s)',fontsize=14,fontweight='bold')
# ax.set_ylabel('magnitude',fontsize=14,fontweight='bold')
# plt.show()


