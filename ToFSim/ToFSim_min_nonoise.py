# Python Imports
import sys
import os
import json
import math
# Library Imports
import numpy as np
# Local Imports
import PlotUtils
import Utils

# Single pixel C-ToF simulation, only photon and read noise are added. No quantization noise.

################# Simulation Parameters ####################
speedOfLight = 2.998e+11 # millimeters per second
maxDepth = 10000 # millimeters
depths = np.arange(0,maxDepth+1)
numPoints = 10001 # number of discrete points representing the mod and demod functions
tau = 2.*maxDepth/speedOfLight # period in seconds
dt = tau / numPoints
times = np.arange(0.,tau,dt)
freq = 1./tau # frequency in hertz
omega = 2.*np.pi*freq # angular frequency in radians per second
albedo = 1.

ambientIllumExpMin = 18
ambientIllumExpMax = 19
ambientIllumFactorMin = 1 # Ambient illumination 1 times stronger than source
ambientIllumFactorMax = 100 # Ambient illumination 10 times stronger than source

readNoiseVarMin = 1
readNoiseVarMax = 100

################# Coding Functions ####################
codingFunction = "SinusoidSinusoid"
k = 3 # Number of measurements
mod = np.zeros((k,numPoints))
demod = np.zeros((k,numPoints))
lookupCurves = np.zeros((k,numPoints))

totalModEnergy = np.sum(0.5 + 0.5*np.cos(omega*times))

for i in range(0,k):
	if codingFunction == "SinusoidSinusoid":
		mod[i,:] = 0.5 + 0.5*np.cos(omega*times) # set sinusoidal coding function
		# mod[i,:] = mod[i,:] / np.sum(mod[i,:]) # normalize so that sum is equal to 1
		demod[i,:] = 0.5 + 0.5*np.cos((omega*times) - 2*i*np.pi/k)
	elif codingFunction == "DeltaSinusoid":
		mod[i,0] = totalModEnergy # set sinusoidal coding function
		demod[i,:] = 0.5 + 0.5*np.cos((omega*times) - 2*i*np.pi/k)
	else:
		sys.exit("Error: Incorrect Coding Function... terminating program")
	
	lookupCurves[i,:] = np.dot(Utils.GenerateCirculantMatrix(mod[i,:]),demod[i,:])

################# Data Parameters ####################

numSamples = 100000
trueDepths = np.random.uniform(low=1.,high=maxDepth,size=numSamples)
readNoiseVar = np.random.uniform(low=readNoiseVarMin,high=readNoiseVarMax,size=numSamples)
ambientIllum = np.random.uniform(low=10^ambientIllumExpMin,high=10^ambientIllumExpMax,size=numSamples)
ambientIllumFactor = np.random.uniform(low=1,high=10,size=numSamples)
dataset = np.zeros((numSamples,k+1))

################ Generate Data #########################
totalAbsDepthError = 0.
for i in range(0,numSamples):
	print "Sample: " + str(i) + " of " + str(numSamples) 

	trueDepth = trueDepths[i]
	shift = (2*trueDepth/speedOfLight)*omega
	shiftedMod = np.zeros((k,numPoints))
	b = np.zeros((k,1))

	for j in range(0,k):
		if codingFunction == "SinusoidSinusoid":
			shiftedMod[j,:] = 0.5 + 0.5*np.cos((omega*times) - shift)
		elif codingFunction == "DeltaSinusoid":
			shiftedMod[j,int(round(trueDepth))] = totalModEnergy
		else:
			sys.exit("Error: Incorrect Coding Function... terminating program")
		
		b[j] = np.dot(shiftedMod[j,:],demod[j,:])

	lookupTable = lookupCurves - b
	# print b
	# print np.sum(np.abs(lookupTable),axis=0)
	# print np.min(np.mean(np.abs(lookupTable),axis=0))
	# print np.argmin(np.sum(np.abs(lookupTable),axis=0))
	# PlotUtils.PlotN(depths,lookupCurves,xlabel='depth',ylabel='brightness',title='lookup curves')

	estimatedDepth = np.argmin(np.sum(np.abs(lookupTable),axis=0))
	totalAbsDepthError = totalAbsDepthError + np.abs(trueDepth - estimatedDepth)
	print "Estimated Depth: " + str(estimatedDepth) + " mm, True Depth: " + str(trueDepth) + " mm" 
	dataset[i,0] = trueDepth
	dataset[i,1:] = np.transpose(b)

meanAbsDepthError = totalAbsDepthError / numSamples
print "Mean Absolute Depth Error: " + str(meanAbsDepthError) + " mm" 

################ Data I/O #########################

datasetJSON = {
	'codingFunction': codingFunction,
	'meanAbsDepthError': meanAbsDepthError,
	'k': k,
	'maxDepth': maxDepth,
	'depthResolution': maxDepth / (numPoints-1),
	'frequency': freq,
	'numSamples': numSamples,
	'noise': 0
}


datasetDirname = codingFunction + str(k)
datasetRootPath = '../Datasets/'
datasetSavePath = datasetRootPath + datasetDirname + '/'

datasetFilename = datasetDirname + '_nonoise'

# print dataset
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
np.savetxt(datasetSavePath + datasetFilename + ".csv",dataset,delimiter=",")
