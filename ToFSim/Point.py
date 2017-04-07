from __future__ import print_function
import numpy as np

class Point(object):
	"""A car for sale by Jeffco Car Dealership.

	Attributes:
		coords: (x,y,z) coordinates. numpy array.
		N: (Nx,Ny,Nz) normal array, i.e orientation of point. numpy array. 
		albedo: (Ax,Ay,Az) albedo components. Range (0-1: [0,255]/255)
	"""

	def __init__(self, coords=np.zeros((1,3),dtype=float), N=np.array([[0,0,1]],dtype=float), albedo=np.ones((1,3),dtype=float)):
		"""Return a new Car object."""
		self.coords = coords
		self.N = N
		self.albedo = albedo

	def __reshape(self):
		self.coords = np.reshape(self.coords,(1,3))
		self.N = np.reshape(self.N,(1,3))
		self.albedo = np.reshape(self.albedo,(1,3))


	def __repr__(self):
		return "Point()"
	
	def __str__(self):
		string = "Point:\n"
		string = string + "    coords (x,y,z): " + str(self.coords) + "\n" 
		string = string + "    normals (Nx,Ny,Nz): " + str(self.N) + "\n" 
		string = string + "    albedo (Ax,Ay,Az): " + str(self.albedo) + "\n" 
		return string

	def getCSVLine(self):
		return str(self.coords[0]) + "," + str(self.coords[1]) + "," + str(self.coords[2]) + "," + str(self.N[0]) + "," + str(self.N[1]) + "," + str(self.N[2]) + "," + str(self.albedo[0]) + "," + str(self.albedo[1]) + "," + str(self.albedo[2])

	def printCSVLine(self):
		print(self.getCSVLine())
