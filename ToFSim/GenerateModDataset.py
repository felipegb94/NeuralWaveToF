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


################### Parameter definition ################### 
modFunction = "Sinusoid"
k = 3
numPoints = 200 # number of points representing the function
speedOfLight = 2.998e+11 # millimeters per second
# Temporal parameters
dMax = 10000 # max depth in millimeters
maxDepth = dMax / 2
dRange = np.linspace(0, dMax, numPoints) # All possible depths that can be recovered
tau = 2*dMax / speedOfLight # period in nanoseconds 
t = np.linspace(0, tau, numPoints)
freq = 1 / (tau) # period in nanoseconds 
omega = 2*np.pi*freq # period in nanoseconds 
dt = tau / numPoints # Time resolution of mod/demod in nanosecond
print("Tau: " + str(tau) + "seconds")

################### Function definitions ###################
depth1=0
depth2=2500
depth3=5000
shift1 = (2. * depth1 / speedOfLight) * (omega)
shift2 = (2. * depth2 / speedOfLight) * (omega)
shift3 = (2. * depth3 / speedOfLight) * (omega)
mod1 = 0.5 + 0.5*np.cos(omega*t - shift1) 
mod2 = 0.5 + 0.5*np.cos(omega*t - shift2) 
mod3 = 0.5 + 0.5*np.cos(omega*t - shift3) 



################### Dataset specs ###################
nDataPoints = 50000

################### Other specs ###################
aveSrcEnergyExponent = 19
aveAmbientEnergyExponent = np.random.uniform(18,19,nDataPoints)
ambientAveEAll = pow(10,aveAmbientEnergyExponent) # ambient illumination strength (avg # of photons)
readNoiseVariance = np.random.uniform(1,50,nDataPoints)
posPoint = np.array([0.,0.,0.])
posLightCamera = np.array([0.,0.,5.])
px = Point(coords=posPoint)
light = LightSource(coords=posLightCamera,aveE=pow(10.0,aveSrcEnergyExponent))
cam = Camera(center=posLightCamera, readNoise=readNoiseVariance[0], fullWellCap=100000)



################ Generate phase shifted functions #########################

trueDepths = np.random.uniform(low=0.,high=maxDepth,size=nDataPoints)
modDataset = np.zeros((nDataPoints,1 + (3 * numPoints)))

for i in range(0,nDataPoints):
	modDataset[i,0] = trueDepths[i]
	shift = (2. * trueDepths[i] / speedOfLight) * (omega)
	# print shift
	# print trueDepths[i]
	if modFunction == "Sinusoid":
		mod = 0.5 + 0.5*np.cos(omega*t - shift)
	elif modFunction == "Delta":
		mod = np.zeros(1,numPoints)
	else:
		sys.exit("Error: Incorrect Coding Function... terminating program")

	# PlotUtils.PlotN(t,Y=np.array([mod]),xlabel='x',ylabel='y',title='plot')
		
	# modDataset[i,1:(numPoints+1)] = mod
	# modDataset[i,(numPoints+1):((2*numPoints)+1)] = mod
	# modDataset[i,((2*numPoints)+1):((3*numPoints)+1)] = mod

	for j in range(0,k):
		# mod[i,0] = 1. # set delta coding function
		# print str((j*numPoints)+1)
		# print str(((j+1)*numPoints)+1)
		modDataset[i,((j*numPoints)+1):(((j+1)*numPoints)+1)] = mod # set sinusoidal coding function



# ################ Save dataset #########################
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
				str(int(numPoints)))

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

# x = dRange * dt
# y = modDataset[0,1:(numPoints+1)]
# print x.shape
# print y.shape

# PlotUtils.PlotN(dt*freq*dRange,Y=np.array([y]),xlabel='x',ylabel='y',title='plot')
# y = modDataset[0,1:numPoints+1]
# PlotUtils.PlotN(t,Y=np.array([y]),xlabel='x',ylabel='y',title='plot')




