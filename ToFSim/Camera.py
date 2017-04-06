from __future__ import print_function
import numpy as np

class Camera(object):
	"""Camera object.

	Attributes:
		center: (x,y,z) coordinates.
		downVector:  
		lookAtVector: 
		nRow: Resolution component
		nCol: Resolution component
		fLengthScaling: Focal length scaling factor
		fLength: focalLength, nRow * nCol * fLengthScaling
		principalPoint: (nRow/2, nCol/2)
		pixelSize:  2e^(-6)
		exposureTime: default - 0.1
		readNoise: default - 10
		fullWellCap: default - 85000
		numBits: default - 14
		gain: fullWellcap / (2^numBits)
		FNumber: default - 2.5
		KMatrix: [ Cam.FocalLength      0        Cam.PrincipalPoint(1);
                 0          Cam.FocalLength   Cam.PrincipalPoint(2);
                 0               0                 1]; 
		RMatrix: [[Cam.LookAtVector(1) Cam.LookAtVector(2) Cam.LookAtVector(3)]';	
				  Cam.DownVector(1)   Cam.DownVector(2)   Cam.DownVector(3)]';
			 	  cross(DownVector, LookAtVector)];
		TMatrix: -RMatrix * center  
	"""

	def __init__(self, center=np.zeros((1,3)), downVector=np.array([[0.,-1.,0.]]), lookAtVector=np.array([[0.,0.,-1.]]), nRow=1, nCol=1, fLengthScaling=3/8, pixelSize=2.*pow(10.,-6), exposureTime=0.1, readNoise=10., fullWellCap=85000, numBits=14, fNumber=2.5):
		"""Return a new camera object."""
		self.center = center
		self.downVector = downVector
		self.lookAtVector = lookAtVector
		self.nRow = nRow
		self.nCol = nCol
		self.fLengthScaling = fLengthScaling
		self.fLength = nRow * nCol * fLengthScaling
		self.principalPoint = np.array((float(nRow)/2,float(nCol)/2))
		self.pixelSize = pixelSize
		self.exposureTime = exposureTime
		self.readNoise = readNoise
		self.fullWellCap = fullWellCap
		self.numBits = numBits
		self.gain = float(fullWellCap) / pow(2,numBits)
		self.fNumber = 2.5
		self.__reshape()
		self.kMat = np.matrix([[self.fLength, 0., self.principalPoint[0]],
								[0., self.fLength, self.principalPoint[1]],
								[0., 0., 1.]])
		self.rMat = np.zeros((3,3))
		self.rMat[:,0] = np.cross(self.downVector,self.lookAtVector)
		self.rMat[:,1] = np.reshape(self.downVector,(3,))
		self.rMat[:,2] = np.reshape(self.lookAtVector,(3,))
		self.tMat = -1*np.matmul(self.rMat.transpose(),self.center.transpose())

	def __reshape(self):
		self.center = np.reshape(self.center,(1,3))
		self.downVector = np.reshape(self.downVector,(1,3))
		self.lookAtVector = np.reshape(self.lookAtVector,(1,3))


	def __repr__(self):
		return "Camera()"

	def __str__(self):
		string = "Camera:\n"
		string = string +  "    center (x,y,z): " + str(self.center) + "\n" 
		string = string + "    resolution (nRow,nCol): " + str(self.nRow*self.nCol) + " (" +str(self.nRow) + ", " + str(self.nCol) + ")" + "\n" 
		string = string + "    focal length: " + str(self.fLength) + "\n" 
		string = string + "    downVector (Ax,Ay,Az): " + str(self.downVector) + "\n" 
		string = string + "    lookAtVector (Ax,Ay,Az): " + str(self.lookAtVector) + "\n" 
		string = string + "    principalPoint (x,y): " + str(self.principalPoint) + "\n" 
		string = string + "    pixel size: " + str(self.pixelSize) + "\n" 
		string = string + "    exposure time: " + str(self.exposureTime) + "\n" 
		string = string + "    readNoise: " + str(self.readNoise) + "\n"
		string = string + "    fullWellcap: " + str(self.fullWellCap) + "\n"
		string = string + "    numBits: " + str(self.numBits) + "\n"
		string = string + "    gain: " + str(self.gain) + "\n"
		string = string + "    fNumber: " + str(self.fNumber) + "\n"
		string = string + "    kMat : " + str(self.kMat) + "\n" 
		string = string + "    rMat : " + str(self.rMat) + "\n" 
		string = string + "    tMat : " + str(self.tMat) + "\n" 


		return string		

