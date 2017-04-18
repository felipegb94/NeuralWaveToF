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

modFunction = "Sinusoid"
k = 3 # Number of measurements
maxDepth = 5000.
dMax = 2 * maxDepth # max depth in millimeter
dSampleMod = 1. # sampling rate in depth values for mod/demod functions
dRange = np.arange(0., dMax+1., dSampleMod) # All possible depths that can be recovered
nDepths = dRange.size
timeRes = 2 * dSampleMod / speedOfLight # Time resolution of mod/demod in seconds
deltaT = timeRes # time resolution is the same as the delta time
tau = nDepths * timeRes 
freq = 1./tau

nDataPoints = 20000

aveSrcEnergyExponent = 19
aveAmbientEnergyExponent = np.random.uniform(18,19,nDataPoints)
ambientAveEAll = pow(10,aveAmbientEnergyExponent) # ambient illumination strength (avg # of photons)
readNoiseVariance = np.random.uniform(1,50,nDataPoints)

posPoint = np.array([0.,0.,0.])
posLightCamera = np.array([0.,0.,5.])
px = Point(coords=posPoint)
light = LightSource(coords=posLightCamera,aveE=pow(10.0,aveSrcEnergyExponent))
cam = Camera(center=posLightCamera, readNoise=readNoiseVariance[0], fullWellCap=100000)



########################### Make coding functions ########################################
# mod = np.zeros((k,nDepths),dtype=float)

if modFunction == "Sinusoid":
	print "Generating Sinusoid Dataset"
	# for i in range(0,k):
	# 	# mod[i,0] = 1. # set delta coding function
	# 	mod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths)) # set sinusoidal coding function
		# mod[i,:] = mod[i,:] / np.sum(mod[i,:]) # normalize so that sum is equal to 1
		# demod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths) - 2*i*np.pi/k)  
# if modFunction == "delta":
# 	for i in range(0,k):
# 		# mod[i,0] = 1. # set delta coding function
# 		mod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths)) # set sinusoidal coding function
# 		mod[i,:] = mod[i,:] / np.sum(mod[i,:]) # normalize so that sum is equal to 1
		# demod[i,:] = 0.5 + 0.5*np.cos((2*np.pi*dRange/nDepths) - 2*i*np.pi/k)  
else:
	sys.exit("Error: Incorrect Coding Function... terminating program")


################ Generate phase shifted functions #########################

trueDepths = np.random.uniform(low=0.,high=maxDepth,size=nDataPoints)
modDataset = np.zeros((nDataPoints,1 + (3 * nDepths)))
for i in range(0,nDataPoints):
	modDataset[i,0] = trueDepths[i]
	shift = (2. * trueDepths[i] / speedOfLight)

	if modFunction == "Sinusoid":
		modIn =  (2*np.pi*freq*deltaT*(dRange)) - (2*np.pi*freq*shift)
		mod = 0.5 + 0.5*np.cos(modIn)
	elif modFunction == "Delta":
		mod = np.zeros(1,nDepths)
	else:
		sys.exit("Error: Incorrect Coding Function... terminating program")

	# PlotUtils.PlotN(deltaT*freq*dRange,Y=np.array([mod]),xlabel='x',ylabel='y',title='plot')
		
	# modDataset[i,1:(nDepths+1)] = mod
	# modDataset[i,(nDepths+1):((2*nDepths)+1)] = mod
	# modDataset[i,((2*nDepths)+1):((3*nDepths)+1)] = mod

	for j in range(0,k):
		# mod[i,0] = 1. # set delta coding function
		# print str((j*nDepths)+1)
		# print str(((j+1)*nDepths)+1)
		modDataset[i,((j*nDepths)+1):(((j+1)*nDepths)+1)] = mod # set sinusoidal coding function



################ Save dataset #########################
datasetJSON = {
	'modFunction': modFunction,
	'k': k,
	'aveSrcEnergyExponent': aveSrcEnergyExponent,
	'aveAmbientEnergyExponent': "aveAmbientEnergyExponent",
	'readNoiseVariance': "readNoiseVariance",
	'fullWellCap': cam.fullWellCap,
	'numBits': cam.numBits,
	'cameraGain': cam.gain,
	'maxDepth': dMax,
	'tau': tau,
	'frequency': freq,
	'nDataPoints': nDataPoints
}

datasetDirName = modFunction + str(k)
datasetReposPath = '../Datasets/'
datasetSavePath = datasetReposPath + datasetDirName + '/'

# datasetFilename = (datasetDirName + '_' + 
# 				str(int(aveAmbientEnergyExponent)) + '_' +
# 				str(int(readNoiseVariance)))


datasetFilename = (datasetDirName + '_' + 
				str(int(dMax)))

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
np.savetxt(datasetSavePath + datasetFilename + ".csv", modDataset,delimiter=",")

x = dRange * deltaT
y = modDataset[0,1:(nDepths+1)]
print x.shape
print y.shape

PlotUtils.PlotN(deltaT*freq*dRange,Y=np.array([y]),xlabel='x',ylabel='y',title='plot')



