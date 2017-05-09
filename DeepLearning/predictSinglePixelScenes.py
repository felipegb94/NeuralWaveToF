from DataUtils import loadEvalDataset
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import pylab
import DataUtils
import keras

# img1 = bAll[1, :, :, :]
# img1 = img1 / np.max(img1)
# true1 = trueDepths[1, :, :, :]

# predict1 = np.zeros(true1.shape)

# for i in range(img1.shape[0]):
#   for j in range(img1.shape[1]):
#     predict1[i, j] = model.predict(np.array([ img1[i, j, :]] ))[0]

# print np.mean(np.abs(true1 - predict1))


sceneIDs = ['Wall','Staircase2','Staircase4','Ramp','HalfSphere']
numSceneIDs = len(sceneIDs)
evalIDs = [0,5000,8900]
numEvalIDs = len(evalIDs)
basePath = "../Datasets/"
(trueDepths,bEval) = DataUtils.loadEvalDataset()

jsonFile = open("results/noisy.json", 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()

model = keras.models.model_from_json(loadedModelJson)
model.load_weights("results/noisy.h5")
print("Loaded model from disk")

model.compile(optimizer='adam',loss='mean_absolute_error')

index = 0
for i in range(0,numSceneIDs):
  for j in range(0,numEvalIDs):
    img = bEval[index, :, :, :]
    img = img / np.max(img)
    estimatedDepths =  np.zeros(trueDepths[index,:,:,:].shape)

    for x in range(img.shape[0]):
      for y in range(img.shape[1]):
        estimatedDepths[x, y] = model.predict(np.array([ img[x, y, :]] ))[0]

    meanAbsErr = np.mean(np.abs(estimatedDepths - trueDepths[index,:,:,:]))
    print "Scene: " + sceneIDs[i] + " depthOffset: " + str(evalIDs[j])
    print "		meanAbsErr = " + str(meanAbsErr)
    np.savetxt("../SinglePixelPredictedScenes/" + sceneIDs[i] + "_" + str(evalIDs[j]) + "_EvalScenes.csv",estimatedDepths,delimiter=",")
    index += 1

# fig = pylab.figure()
# ax = Axes3D(fig)

# xs, ys, zs = [], [], []
# for i in range(img1.shape[0]):
#   for j in range(img1.shape[1]):
#     xs.append(i)
#     ys.append(j)
#     zs.append(predict1[i, j])

# ax.scatter(xs, ys, zs)
# pyplot.show()
