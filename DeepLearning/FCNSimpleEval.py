import numpy as np
import DataUtils
import keras


sceneIDs = ['Wall','Staircase2','Staircase4','Ramp','HalfSphere']
numSceneIDs = len(sceneIDs)
evalIDs = [0,5000,8900]
numEvalIDs = len(evalIDs)
basePath = "../Datasets/"
numSamples = numSceneIDs*numEvalIDs
(trueDepths,bEval) = DataUtils.loadEvalDataset()

jsonFile = open("FCNSimple.json", 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()

model = keras.models.model_from_json(loadedModelJson)
model.load_weights("FCNSimple.h5")
print("Loaded model from disk")

model.compile(optimizer='adam',loss='mean_absolute_error')

estimatedDepths = model.predict(bEval)
print estimatedDepths.shape

index = 0
for i in range(0,numSceneIDs):
	for j in range(0,numEvalIDs):
		meanAbsErr = np.mean(np.abs(estimatedDepths[index,:,:,:] - trueDepths[index,:,:,:]))
		print "Scene: " + sceneIDs[i] + " depthOffset: " + str(evalIDs[j])
		print "		meanAbsErr = " + str(meanAbsErr)
		np.savetxt("../FCNSimplePredictedScenes/" + sceneIDs[i] + "_" + str(evalIDs[j]) + "_EvalScenes.csv",estimatedDepths[index,:,:,0],delimiter=",")
		index = index + 1