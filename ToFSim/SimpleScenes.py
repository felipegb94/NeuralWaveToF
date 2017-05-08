import math
import numpy as np
from plyfile import PlyData, PlyElement

def GetPlyElement(x,y,z,albedo):
	return np.array((float(x),float(y),float(z),float(z),float(albedo)),dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])

def WallScene(numRows, numCols, depth, albedo):
	SceneID = 'WallScene'
	# Create depth 2D matrix
	depthMap = np.ones((numRows,numCols))*depth	
	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])
	index = 0
	for i in range(0,numRows):
		for j in range(0,numCols):
			x = float(j)/float(numCols)
			y = float(i)/float(numRows)
			pcArray[index] = GetPlyElement(x,y,depth,albedo)
			index = index + 1

	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write('../GroundTruthScenes/'+SceneID + '.ply')
	np.savetxt('../GroundTruthScenes/'+SceneID + '.csv',depthMap,delimiter=",")

	return (depthMap,plyEl)

def Staircase2Scene(numRows, numCols, depths, albedo):
	SceneID = 'Staircase2'
	# Create depth 2D matrix
	depthMap = np.zeros((numRows,numCols))	
	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])
	index = 0
	for i in range(0,int(math.floor(numRows/2))):
		for j in range(0,numCols):
			x = float(j)/float(numCols)
			y = float(i)/float(numRows)
			pcArray[index] = GetPlyElement(x,y,depths[0],albedo)
			index = index + 1
			depthMap[i,j] = depths[0]

	for i in range(int(math.floor(numRows/2)),numRows):
		for j in range(0,numCols):
			x = float(j)/float(numCols)
			y = float(i)/float(numRows)
			pcArray[index] = GetPlyElement(x,y,depths[1],albedo)
			index = index + 1
			depthMap[i,j] = depths[1]


	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write('../GroundTruthScenes/'+SceneID + '.ply')
	np.savetxt('../GroundTruthScenes/'+SceneID + '.csv',depthMap,delimiter=",")

	return (depthMap,plyEl)

def Staircase4Scene(numRows, numCols, depths, albedo):
	SceneID = 'Staircase4'
	# Create depth 2D matrix
	depthMap = np.zeros((numRows,numCols))	
	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])
	index = 0

	for stairID in range(0,4):
		for i in range(stairID*int(math.floor(numRows/4)),(stairID+1)*int(math.floor(numRows/4))):
			for j in range(0,numCols):
				x = float(j)/float(numCols)
				y = float(i)/float(numRows)
				pcArray[index] = GetPlyElement(x,y,depths[stairID],albedo)
				index = index + 1
				depthMap[i,j] = depths[stairID]


	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write('../GroundTruthScenes/'+SceneID + '.ply')
	np.savetxt('../GroundTruthScenes/'+SceneID + '.csv',depthMap,delimiter=",")

	return (depthMap,plyEl)



def RampScene(numRows, numCols, slope, centerDepth, albedo):
	SceneID = 'RampScene'
	# Create depth 2D matrix
	depthMap = np.zeros((numRows,numCols))	
	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])
	index = 0
	for i in range(0,numRows):
		for j in range(0,numCols):
			x = float(j)/float(numCols)
			y = float(i)/float(numRows)
			z = centerDepth + (y*slope)
			depthMap[i,j] = z
			pcArray[index] = GetPlyElement(x,y,z,albedo)
			index = index + 1


	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write('../GroundTruthScenes/'+SceneID + '.ply')
	np.savetxt('../GroundTruthScenes/'+SceneID + '.csv',depthMap,delimiter=",")
	return (depthMap,plyEl)


def HalfSphereScene(numRows, numCols, radius, centerDepth, albedo):
	SceneID = 'HalfSphereScene'
	# Create depth 2D matrix
	depthMap = np.zeros((numRows,numCols))	
	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])
	index = 0
	for i in range(0,numRows):
		for j in range(0,numCols):
			x = float(2*j)/float(numCols)
			y = float(2*i)/float(numRows)
			xterm = x - 1.#(float(numCols)/2.) 
			yterm = y - 1.#(float(numRows)/2.)
			rootTerm = (radius*radius) - (xterm*xterm) - (yterm*yterm) 
			if(rootTerm >= 0):
				z = np.sqrt(rootTerm) + centerDepth
				depthMap[i,j] = z
				pcArray[index] = GetPlyElement(x,y,z,albedo)
				index = index + 1
			else:
				depthMap[i,j] = centerDepth
				pcArray[index] = GetPlyElement(x,y,centerDepth,albedo)
				index = index + 1


	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write('../GroundTruthScenes/'+SceneID + '.ply')
	np.savetxt('../GroundTruthScenes/'+SceneID + '.csv',depthMap,delimiter=",")
	return (depthMap,plyEl)