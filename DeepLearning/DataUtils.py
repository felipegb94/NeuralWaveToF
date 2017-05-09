import numpy as np

def loadDataset():
	numDatasets = 5
	datasetPath1 = '../Datasets/WallSinusoidSinusoid3/WallSinusoidSinusoid3_noisy0.npy'
	datasetPath2 = '../Datasets/RampSinusoidSinusoid3/RampSinusoidSinusoid3_noisy0.npy'
	datasetPath3 = '../Datasets/Staircase2SinusoidSinusoid3/Staircase2SinusoidSinusoid3_noisy0.npy'
	datasetPath4 = '../Datasets/Staircase4SinusoidSinusoid3/Staircase4SinusoidSinusoid3_noisy0.npy'
	datasetPath5 = '../Datasets/HalfSphereSinusoidSinusoid3/HalfSphereSinusoidSinusoid3_noisy0.npy'
	dataset1 = np.load(datasetPath1)
	dataset2 = np.load(datasetPath2)
	dataset3 = np.load(datasetPath3)
	dataset4 = np.load(datasetPath4)
	dataset5 = np.load(datasetPath5)
	(kp1, numRows, numCols, numSamples1) = dataset1.shape
	k = kp1-1
	numSamples = numDatasets*numSamples1

	bAll = np.zeros((numSamples,numRows,numCols,k))
	trueDepths = np.zeros((numSamples,numRows,numCols,1))
	index = 0
	for i in range(0,numSamples1):
		trueDepths[index,:,:,:] = dataset1[0,:,:,i].reshape((numRows,numCols,1))
		bAll[index,:,:,0] = dataset1[1,:,:,i]
		bAll[index,:,:,1] = dataset1[2,:,:,i]
		bAll[index,:,:,2] = dataset1[3,:,:,i]
		trueDepths[index+1,:,:,:] = dataset2[0,:,:,i].reshape((numRows,numCols,1))
		bAll[index+1,:,:,0] = dataset2[1,:,:,i]
		bAll[index+1,:,:,1] = dataset2[2,:,:,i]
		bAll[index+1,:,:,2] = dataset2[3,:,:,i]
		trueDepths[index+2,:,:,:] = dataset3[0,:,:,i].reshape((numRows,numCols,1))
		bAll[index+2,:,:,0] = dataset3[1,:,:,i]
		bAll[index+2,:,:,1] = dataset3[2,:,:,i]
		bAll[index+2,:,:,2] = dataset3[3,:,:,i]
		trueDepths[index+3,:,:,:] = dataset4[0,:,:,i].reshape((numRows,numCols,1))
		bAll[index+3,:,:,0] = dataset4[1,:,:,i]
		bAll[index+3,:,:,1] = dataset4[2,:,:,i]
		bAll[index+3,:,:,2] = dataset4[3,:,:,i]
		trueDepths[index+4,:,:,:] = dataset5[0,:,:,i].reshape((numRows,numCols,1))
		bAll[index+4,:,:,0] = dataset5[1,:,:,i]
		bAll[index+4,:,:,1] = dataset5[2,:,:,i]
		bAll[index+4,:,:,2] = dataset5[3,:,:,i]
		index = index+numDatasets


	bAll = bAll / np.max(bAll)

	numTrainSamples = int(round(0.8*numSamples))
	numTuneSamples = int(round(0.1*numSamples))
	numTestSamples = int(round(0.1*numSamples))
	bTrain = bAll[0:numTrainSamples,:,:,:]
	bTune = bAll[numTrainSamples:numTrainSamples+numTuneSamples,:,:,:]
	bTest = bAll[numTrainSamples+numTuneSamples:numSamples,:,:,:]
	trueDepthsTrain = trueDepths[0:numTrainSamples,:,:,:]
	trueDepthsTune = trueDepths[numTrainSamples:numTrainSamples+numTuneSamples,:,:,:]
	trueDepthsTest = trueDepths[numTrainSamples+numTuneSamples:numSamples,:,:,:]
	print "bAll shape (input): "
	print bAll.shape
	print "bTrain shape (input): "
	print bTrain.shape
	print "bTune shape (input): "
	print bTune.shape
	print "bTest shape (input): "
	print bTest.shape
	print "trueDepths shape (output): "
	print trueDepths.shape
	print "trueDepthsTrain shape (input): "
	print trueDepthsTrain.shape
	print "trueDepthsTune shape (input): "
	print trueDepthsTune.shape
	print "trueDepthsTest shape (input): "
	print trueDepthsTest.shape

	return (trueDepthsTrain, bTrain, trueDepthsTune, bTune, trueDepthsTest, bTest)


def loadEvalDataset():
	sceneIDs = ['Wall','Staircase2','Staircase4','Ramp','HalfSphere']
	numSceneIDs = len(sceneIDs)
	evalIDs = [0,5000,8900]
	numEvalIDs = len(evalIDs)
	basePath = "../Datasets/"
	numSamples = numSceneIDs*numEvalIDs
	(numRows,numCols) = (128,128)
	bAll = np.zeros((numSamples,numRows,numCols,3))
	trueDepths = np.zeros((numSamples,numRows,numCols,1))

	index = 0
	for i in range(0,numSceneIDs):
		datasetPath = basePath + sceneIDs[i] + "SinusoidSinusoid3/" + sceneIDs[i] + "SinusoidSinusoid3_EvalScenes.npy"
		dataset = np.load(datasetPath)
		numDatasetSamples = dataset.shape[3]
		for j in range(0,numDatasetSamples):
			trueDepths[index,:,:,:] = dataset[0,:,:,j].reshape((numRows,numCols,1))	   			
			bAll[index,:,:,0] = dataset[1,:,:,j]	   			
			bAll[index,:,:,1] = dataset[2,:,:,j]	   			
			bAll[index,:,:,2] = dataset[3,:,:,j]	   			
			index = index + 1
	
	bAll = bAll / np.max(bAll)

	print "bAll shape (input): "
	print bAll.shape
	print "trueDepths shape (output): "
	print trueDepths.shape


	return (trueDepths,bAll)

