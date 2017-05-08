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
import SimpleScenes

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
ambientIllumFactorMin = 0 # Ambient illumination 1 times stronger than source
ambientIllumFactorMax = 5 # Ambient illumination 10 times stronger than source

readNoiseVarMin = 1
readNoiseVarMax = 100

################# Coding Functions ####################
codingFunction = "SinusoidSinusoid"
k = 3 # Number of measurements
mod = np.zeros((k,numPoints))
demod = np.zeros((k,numPoints))
baseLookupCurves = np.zeros((k,numPoints))

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
	baseLookupCurves[i,:] = np.dot(Utils.GenerateCirculantMatrix(mod[i,:]),demod[i,:])

demodS = np.sum(demod, axis=1).reshape((k,1))
print demodS.shape
################# Data Parameters ####################

numSampleScenes = 1000
sceneID = "HalfSphere"
sceneRows = 128
sceneCols = 128

if sceneID == "Wall":
	(depthMapBase,plyBase) = SimpleScenes.WallScene(sceneRows,sceneCols,0.,albedo)   
	depthMapBase = depthMapBase*1000 # Transform to mm  
	depthOffsets = np.random.uniform(low=0,high=maxDepth-1000,size=numSampleScenes)
elif sceneID == "HalfSphere":
	radius = 1.
	(depthMapBase,plyBase) = SimpleScenes.HalfSphereScene(sceneRows,sceneCols,radius,0.,albedo)
	depthMapBase = depthMapBase*1000 # Transform to mm  
	depthOffsets = np.random.uniform(low=0,high=maxDepth-1000,size=numSampleScenes)
elif sceneID == "Staircase2":
	depths = [0,0.25]
	(depthMapBase,plyBase) = SimpleScenes.Staircase2Scene(sceneRows,sceneCols,depths,albedo)   
	depthMapBase = depthMapBase*1000 # Transform to mm  
	depthOffsets = np.random.uniform(low=0,high=maxDepth-1000,size=numSampleScenes)
elif sceneID == "Staircase4":
	depths = [0.,0.25,0.5,0.75]
	(depthMapBase,plyBase) = SimpleScenes.Staircase4Scene(sceneRows,sceneCols,depths,albedo)   
	depthMapBase = depthMapBase*1000 # Transform to mm  
	depthOffsets = np.random.uniform(low=0,high=maxDepth-1000,size=numSampleScenes)
elif sceneID == "Ramp":
	slope = 1.
	(depthMapBase,plyBase) = SimpleScenes.RampScene(sceneRows,sceneCols,slope,0.,albedo)   
	depthMapBase = depthMapBase*1000 # Transform to mm  
	depthOffsets = np.random.uniform(low=0,high=maxDepth-1000,size=numSampleScenes)
else:
	sys.exit("Error: Incorrect SceneID... terminating program")


depthMapsAll = np.zeros((sceneRows,sceneCols,numSampleScenes))
print np.min(depthMapsAll)
print np.max(depthMapsAll)


readNoiseVar = np.random.uniform(low=readNoiseVarMin,high=readNoiseVarMax,size=numSampleScenes)
ambientIllum = np.random.uniform(low=10^ambientIllumExpMin,high=10^ambientIllumExpMax,size=numSampleScenes)
ambientIllumFactor = np.random.uniform(low=ambientIllumFactorMin,high=ambientIllumFactorMax,size=numSampleScenes)

for i in range(0,numSampleScenes):
	depthMapsAll[:,:,i] = depthMapBase + depthOffsets[i]		

############### Generate Data #########################
totalAbsDepthError = 0.
perSceneMeanAbsDepthError = np.zeros((numSampleScenes,)) 
brightnessAll1 = np.zeros((sceneRows,sceneCols,numSampleScenes))
brightnessAll2 = np.zeros((sceneRows,sceneCols,numSampleScenes))
brightnessAll3 = np.zeros((sceneRows,sceneCols,numSampleScenes))

for i in range(0,numSampleScenes):
	print sceneID + " Sample: " + str(i) + " of " + str(numSampleScenes) 
	### Get all ambient parameters for the scene
	ambientIllum = ambientIllumFactor[i]
	readNoise = readNoiseVar[i]*np.random.normal(0,1)
	lookupCurves = np.zeros((k,numPoints))
	gamma = ambientIllum*demodS
	# Construct lookup table for specific ambient illumination
	for j in range(0,k):
		lookupCurves[j,:] = baseLookupCurves[j,:] + gamma[j]

	### Iterate through each pixel in the scene to compute the depth
	currScene = depthMapsAll[:,:,i]
	for currRow in range(0,sceneRows):
		for currCol in range(0,sceneRows):

			trueDepth = currScene[currRow,currCol]
			shiftedMod = np.zeros((1,numPoints))
			shift = (2*trueDepth/speedOfLight)*omega

			if codingFunction == "SinusoidSinusoid":
				shiftedMod = 0.5 + 0.5*np.cos((omega*times) - shift) + (ambientIllum)
			elif codingFunction == "DeltaSinusoid":
				shiftedMod = totalModEnergy + (ambientIllum)
			else:
				sys.exit("Error: Incorrect Coding Function... terminating program")

			b = np.zeros((k,1))
			for j in range(0,k):
				b[j] = np.dot(shiftedMod,demod[j,:])
				photonNoiseVar = np.sqrt(b[j])
				photonNoise = photonNoiseVar*np.random.normal(0,1)
				b[j] = b[j] + readNoise + photonNoise

			lookupTable = lookupCurves - b

			estimatedDepth = np.argmin(np.sum(np.abs(lookupTable),axis=0))
			perSceneMeanAbsDepthError[i] = 0.5*(perSceneMeanAbsDepthError[i] + np.abs(trueDepth - estimatedDepth))
			# print "Estimated Depth: " + str(estimatedDepth) + " mm, True Depth: " + str(trueDepth) + " mm" 
			brightnessAll1[currRow,currCol,i] = b[0]
			brightnessAll2[currRow,currCol,i] = b[1]
			brightnessAll3[currRow,currCol,i] = b[2]
	print "Mean Absolute Depth Error: " + str(perSceneMeanAbsDepthError[i])


# # ################ Data I/O #########################

datasetJSON = {
	'codingFunction': codingFunction,
	'perSceneMeanAbsDepthError': perSceneMeanAbsDepthError.reshape((numSampleScenes,)).tolist(),
	'datasetMeanAbsDepthError': np.mean(perSceneMeanAbsDepthError),
	'k': k,
	'maxDepth': maxDepth,
	'depthResolution': maxDepth / (numPoints-1),
	'frequency': freq,
	'numSampleScenes': numSampleScenes,
}

dataset = np.array([(depthMapsAll),(brightnessAll1),(brightnessAll2),(brightnessAll3)])

ID = 0
datasetDirname = sceneID + codingFunction + str(k)
datasetRootPath = '../Datasets/'
datasetSavePath = datasetRootPath + datasetDirname + '/'

datasetFilename = datasetDirname + '_noisy' + str(ID)

# print dataset
# See if dataset directory exists, if not create it
try: 
	os.makedirs(datasetSavePath)
except OSError:
	if not os.path.isdir(datasetSavePath):
		raise
# # Save dataset information into JSON file
with open(datasetSavePath + datasetFilename + ".json", "w") as outfile:
	json.dump(datasetJSON, outfile, indent=4)
# Save dataset CSV file
np.save(datasetSavePath + datasetFilename,dataset)


