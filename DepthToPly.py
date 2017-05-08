import math
import numpy as np
from plyfile import PlyData, PlyElement

def GetPlyElement(x,y,z,albedo):
	return np.array((float(x),float(y),float(z),float(z),float(albedo)),dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])

def DepthToPly(depthMapPath,depthMapFilename):
	albedo = 1.
	depthMap = np.loadtxt(depthMapPath+depthMapFilename+".csv",delimiter=",")

	if np.max(depthMap) > 20:
		depthMap = depthMap / 1000.

	(numRows,numCols) = depthMap.shape

	# Create ply object
	numPoints = numRows*numCols
	pcArray = np.zeros((numPoints,), dtype=[('x','float'),('y','float'),('z','float'),('depth','float'),('albedo','float')])

	index = 0

	for i in range(0,numRows):
		for j in range(0,numCols):
			x = float(j)/float(numCols)
			y = float(i)/float(numRows)
			z = depthMap[i][j]
			pcArray[index] = GetPlyElement(x,y,z,albedo)
			index = index + 1


	plyEl = PlyElement.describe(pcArray,'vertex')
	PlyData([plyEl],text=True).write(depthMapPath+depthMapFilename+'.ply')


sceneIDs = ['Wall','Staircase2','Staircase4','Ramp','HalfSphere']
numSceneIDs = len(sceneIDs)
evalIDs = [0,5000,8900]
numEvalIDs = len(evalIDs)
basePath = "../Datasets/"


depthMapPath = "AnalyticalScenes/"
depthMapPath = "GroundTruthScenes/"
# depthMapFilename = "Wall_0_EvalScenes"

for i in range(0,numSceneIDs):
	for j in range(0,numEvalIDs):
		print "Scene: " + sceneIDs[i] + " depthOffset: " + str(evalIDs[j])
		depthMapFilename = sceneIDs[i] + "_" + str(evalIDs[j]) + "_EvalScenes"
		DepthToPly(depthMapPath,depthMapFilename)




