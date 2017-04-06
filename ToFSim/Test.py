import numpy as np
from Point import Point
from LightSource import LightSource
from Camera import Camera
import PlotUtils
import Utils

speedOfLight = 2.998e+11 # millimeters per second

k = 3 # Number of measurements
dMax = 10000 # max depth in millimeter
dSampleMod = 1 # sampling rate in depth values for mod/demod functions
dRange = np.arange(0, dMax+1, dSampleMod) # All possible depths that can be recovered
nDepths = dRange.size
timeRes = 2 * dSampleMod / speedOfLight # Time resolution of mod/demod in seconds
print(dRange)
print(nDepths)

ambientAveE = pow(10,19) # ambient illumination strength (avg # of photons)

posPoint = np.array([0.,0.,0.])
posLightCamera = np.array([0.,0.,5.])
px = Point(coords=posPoint)
light = LightSource(coords=posLightCamera)
cam = Camera(center=posLightCamera)
# print px
# print light
# print cam

distPointLight = pow(np.sum(np.power(px.coords - light.coords,2)),0.5) # Distance between point and light
distTrue = distPointLight
cosTheta = np.dot(light.coords - px.coords, px.N.transpose()) / distPointLight # dot product between scene normals and the line joining light source and scene point. Normalize by distance

betaMat = np.zeros((1,1,3))
betaMat[0,0,:] = 1 / (pow(distPointLight,2))
betaMat[0,0,:] = betaMat[0,0,:] * cosTheta # multiply all elems by cosTheta
betaMat[0,0,:] = np.multiply(betaMat[0,0,:], px.albedo/np.pi) # element-wise multiplication
betaMat[0,0,:] = betaMat[0,0,:] * (np.pi/4) / pow(cam.fNumber,2) 
betaMat[0,0,:] = betaMat[0,0,:] * pow(cam.pixelSize,2) 
betaMat = betaMat.clip(min=0) # make all negative entries 0

chiMat = np.zeros((1,1,3))
# Including the effect of shading and albedo and conversion from irradiance to radiance assuming Lambertian surface (assuming ambient source is at infinity so no foreshortening, but the same direction as light source, so some costheta effect)
chiMat[0,0,:]  = cosTheta * px.albedo / np.pi;     
#Including the effect of lens aperture during conversion from scene radiance to camera irradiance. Assuming scene angle (angle made by line joining pixel to scene patch with the optical axis) is 0 so that cos(scene Angle)^4 = 1 (paraxial assumption)
chiMat[0,0,:]  = chiMat[0,0,:] * (np.pi/4) / pow(cam.fNumber,2);   
# Converting irradiance to flux (multiply by pixel area)                        
chiMat[0,0,:]  = chiMat[0,0,:] * pow(cam.pixelSize,2);

########################### Make coding functions ########################################
mod = np.zeros((k,nDepths),dtype=float)
demod = np.zeros((k,nDepths),dtype=float)

for i in range(0,k):
	mod[i,0] = 1. # set delta coding function
	mod[i,:] = mod[i,:] / np.sum(mod[i,:]) # normalize so that sum is equal to 1
	demod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths) - 2*i*np.pi/k)  

print demod
print mod

# PlotUtils.PlotN(dRange*timeRes,mod,xlabel='time',ylabel='exposure',title='modulation codes')
# PlotUtils.PlotN(dRange*timeRes,demod,xlabel='time',ylabel='exposure',title='demodulation codes')

########################### Compute Intensities ########################################




