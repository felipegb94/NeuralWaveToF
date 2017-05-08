# Python Imports
import sys
import os
import json
# Library Imports
import numpy as np
# Local Imports
from Point import Point
from LightSource import LightSource
from Camera import Camera
import PlotUtils
import Utils


speedOfLight = 2.998e+11 # millimeters per second

codingFunction = "SinusoidSinusoid"
k = 3 # Number of measurements
maxDepth = 5000
dMax = 2 * maxDepth # max depth in millimeter
dSampleMod = 1 # sampling rate in depth values for mod/demod functions
dRange = np.arange(0, dMax+1, dSampleMod) # All possible depths that can be recovered
nDepths = dRange.size
timeRes = 2 * dSampleMod / speedOfLight # Time resolution of mod/demod in seconds
deltaT = timeRes # time resolution is the same as the delta time
print(dRange)
print(nDepths)

nDataPoints = 100000

aveSrcEnergyExponent = 19
aveAmbientEnergyExponent = np.random.uniform(18,19,nDataPoints)
ambientAveEAll = pow(10,aveAmbientEnergyExponent) # ambient illumination strength (avg # of photons)
readNoiseVariance = np.random.uniform(1,50,nDataPoints)

posPoint = np.array([0.,0.,0.])
posLightCamera = np.array([0.,0.,5.])
px = Point(coords=posPoint)
light = LightSource(coords=posLightCamera,aveE=pow(10.0,aveSrcEnergyExponent))
cam = Camera(center=posLightCamera, readNoise=readNoiseVariance[0], fullWellCap=100000)


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

if codingFunction == "SinusoidSinusoid":
	for i in range(0,k):
		# mod[i,0] = 1. # set delta coding function
		mod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths)) # set sinusoidal coding function
		mod[i,:] = mod[i,:] / np.sum(mod[i,:]) # normalize so that sum is equal to 1
		demod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths) - 2*i*np.pi/k)  
else:
	sys.exit("Error: Incorrect Coding Function... terminating program")
# PlotUtils.PlotN(dRange*timeRes,mod,xlabel='time',ylabel='exposure',title='modulation codes')
# PlotUtils.PlotN(dRange*timeRes,demod,xlabel='time',ylabel='exposure',title='demodulation codes')

########################### Compute Intensities ########################################

modPeriodEffective = nDepths * timeRes
nPeriods = cam.exposureTime / modPeriodEffective  # number of periods we integrate for
mod = mod * light.aveE * modPeriodEffective / timeRes # scale mod so that it is in # photons 

grayIMat = np.zeros((cam.nRows,cam.nCols,k))

trueDists = np.random.uniform(low=0.,high=maxDepth,size=nDataPoints)
print trueDists
bMeasurements = np.zeros((nDataPoints,k))

for i in range(0,k):

	currMod = mod[i,:]
	currDemod = demod[i,:]
	circulantMod = Utils.GenerateCirculantMatrix(currMod)
	correlation = np.matmul(Utils.GenerateCirculantMatrix(currMod),currDemod) * timeRes

	for dIndex in range(0,nDataPoints):
		ambientAveE = ambientAveEAll[dIndex]
		cam.readNoise = readNoiseVariance[dIndex]

		# Table lookup: multiply distance by 1000 to transform to millimeters
		# Find index i s.t dRange(i)==distPointLight*1000, and get the correlation value at i. 
		# We will have one sample per pixel.
		# corrFunctionSample = np.interp(distPointLight * 1000, dRange, correlation)
		corrFunctionSample = np.interp(trueDists[dIndex], dRange, correlation)

		kappa = sum(currDemod) * timeRes # Integral of demodulation function over 1 period

		bVals = np.zeros(betaMat.shape)
		for j in range(0, bVals.shape[2]):
			bVals[:,:,j] = (nPeriods) * (np.multiply(betaMat[:,:,j], corrFunctionSample) + 
						(ambientAveE * kappa * chiMat[:,:,j]))
		#### end bVals for loop

		# print bVals

	####################### Applying perspective projection ################################

	# Basically fit all the scene into the NRox X nCol pixel matrix in the camera
	# Take into account that multiple pixel intensities will be mapping into multiple pixels...
	# Not needed for a single point/pixel setup b.c we are just mapping that point to that pixel.

		iVals = bVals

	####################### Applying noise ################################

		# generate photon noise, one for each rgb channel
		photonNoiseVariance = np.sqrt(iVals)
		photonNoise = np.multiply(photonNoiseVariance, np.random.normal(0,1,iVals.shape))
		# generate read noise, one for each rgb channel
		readNoise = np.multiply(cam.readNoise,np.random.normal(0,1,iVals.shape))
		noise = photonNoise + readNoise

		iMatTmp = iVals + noise
		fullWellIndeces = np.where(iMatTmp >= cam.fullWellCap)
		if(len(fullWellIndeces[0]) != 0):
			print iMatTmp
			print(iMatTmp[fullWellIndeces])
			# sys.exit("Error wellCap is FULL! ")


		# Computing noisy intensity, and applying bounds
		iMat = np.maximum(np.minimum(iVals + noise, cam.fullWellCap), 0)



		# Applying gain, and quantization
		iMat = iMat / cam.gain;               

		# converting photons to digital number
		iMat = np.round(iMat) / pow(2,cam.numBits); 
		# print iMat

		# convert to grayscale
		bMeasurements[dIndex,i] = (iMat[:,:,0] / 3.) + (iMat[:,:,1] / 3.) + (iMat[:,:,2] / 3.)
		# print (iMat[:,:,0] * 0.3) + (iMat[:,:,1] * 0.59) + (iMat[:,:,2] * 0.11)
		# grayIMat[:,:,i] = (iMat[:,:,0] * 0.3) + (iMat[:,:,1] * 0.59) + (iMat[:,:,2] * 0.11)

print bMeasurements
print bMeasurements.shape
print trueDists
print trueDists.shape
print trueDists.reshape((nDataPoints,1)).shape
trueDists = trueDists.reshape((nDataPoints,1))

output = np.concatenate((trueDists,bMeasurements),axis=1)
print output 
print output.shape 


################ Save dataset #########################
datasetJSON = {
	'codingFunction': codingFunction,
	'k': k,
	'aveSrcEnergyExponent': aveSrcEnergyExponent,
	'aveAmbientEnergyExponent': "aveAmbientEnergyExponent",
	'readNoiseVariance': "readNoiseVariance",
	'fullWellCap': cam.fullWellCap,
	'numBits': cam.numBits,
	'cameraGain': cam.gain,
	'maxDepth': dMax,
	'nDataPoints': nDataPoints
}

datasetDirName = codingFunction + str(k)
datasetReposPath = '../Datasets/'
datasetSavePath = datasetReposPath + datasetDirName + '/'

# datasetFilename = (datasetDirName + '_' + 
# 				str(int(aveAmbientEnergyExponent)) + '_' +
# 				str(int(readNoiseVariance)))


datasetFilename = (datasetDirName + '_' + 
				'rand' + '_' + 'rand')

# See if dataset directory exists, if not create it
try: 
	os.makedirs(datasetSavePath)
except OSError:
	if not os.path.isdir(datasetSavePath):
		raise

# Save dataset information into JSON file
with open(datasetSavePath + datasetFilename + ".json", "w") as outfile:
	json.dump(datasetJSON, outfile, indent=4)
# Save dataset CSV file
np.savetxt(datasetSavePath + datasetFilename + ".csv",output,delimiter=",")

# a = np.array([trueDists,])

################ Validate recovered depth #########################
# dist = trueDists[0]
# B = bMeasurements[0,:]
# C = np.matrix([ [1,np.cos(0),np.sin(0)],
# 				[1,np.cos(2*np.pi/3),np.sin(2*np.pi/3)],
# 				[1,np.cos(4*np.pi/3),np.sin(4*np.pi/3)]
# 			  ])
# X = np.linalg.solve(C, B)
# phi = np.arccos(X[1]/(np.sqrt((X[1]*X[1]) + (X[2]*X[2]))))
# frequency = 1/modPeriodEffective
# omega = 2*np.pi*frequency
# print "phi = {}".format(phi)
# print "real depth = {}".format(dist)
# print "recovered depth = {}".format(phi*speedOfLight/(2*omega))


# output = np.array([np.reshape(trueDists),bMeasurements])
# print output
### end modulation for loop
