import numpy as np
import math

# This script calculates the mean absolute difference error between the calculated depth and true depth

# Parameters to calculate the frequency and depth ranges
speedOfLight = 2.998e+11 # millimeters per second
dMax = 10000 # max depth in millimeter
dSampleMod = 1 # sampling rate in depth values for mod/demod functions
dRange = np.arange(0, dMax+1, dSampleMod) # All possible depths that can be recovered
nDepths = dRange.size
timeRes = 2 * dSampleMod / speedOfLight
modPeriodEffective = nDepths * timeRes
frequency = 1/modPeriodEffective
omega = 2*np.pi*frequency

codingFunction = "SinusoidSinusoid"
k = 3 # Number of measurements

# Input data
datasetDirName = codingFunction + str(k)
# filename = datasetDirName + "/" + datasetDirName + "_" + str(19) + "_" + str(50) + ".csv" 
filename = datasetDirName + "/" + datasetDirName + "_rand_rand.csv"

data = np.loadtxt(filename, delimiter=',')
trueDepths = data[:,0]

brightnessMeasurements = data[0:10000,1:4]
(N, k) = brightnessMeasurements.shape

print "Test set size: "  + str(N)

C = np.matrix([ 
				[1, np.cos(0),			np.sin(0)],
				[1, np.cos(2*np.pi/3),	np.sin(2*np.pi/3)],
				[1, np.cos(4*np.pi/3),	np.sin(4*np.pi/3)]
			  ])

sumAbsDiff = 0.
sumSqDiff = 0.

for i in range(0,N):
	B = brightnessMeasurements[i,:]
	X = np.linalg.solve(C, B)
	phi = np.arccos(X[1]/(np.sqrt((X[1]*X[1]) + (X[2]*X[2]))))
	estimatedDepth = phi*speedOfLight/(2*omega)
	diff = (trueDepths[i]-estimatedDepth)
	absDiff = math.fabs(diff)
	sqDiff = (diff)*diff
	sumAbsDiff = sumAbsDiff + absDiff
	sumSqDiff = sumSqDiff + sqDiff
	# print "phi = {}".format(phi)
	# print "real depth = {}".format(trueDepths[i])
	# print "recovered depth = {}".format(estimatedDepth)	
	# print "absolute difference = {}".format(absDiff)

print "mean absolute difference = {}".format(sumAbsDiff/N)
print "mean square difference = {}".format(sumSqDiff/N)
