from __future__ import print_function
import numpy as np

class LightSource(object):
	"""A car for sale by Jeffco Car Dealership.

	Attributes:
		coords: (x,y,z) coordinates. numpy array.
		aveE: Light source strength (average no. of photons emitted by the source per second over the entire scene)
	"""

	def __init__(self, coords=np.zeros((1,3)),aveE=pow(10.0,19)):
		"""Return a new Car object."""
		self.coords = coords
		self.aveE = aveE

		self.__reshape() # Make sure vector dimensions are correct

	def __reshape(self):
		self.coords = np.reshape(self.coords,(1,3))

	def __repr__(self):
		return "LightSource()"
	
	def __str__(self):
		string = "LightSource:\n"
		string = string + "    coords (x,y,z): " + str(self.coords) + "\n" 
		string = string + "    aveE: " + str(self.aveE) + "\n" 
		return string

	def getCSVLine(self):
		return str(self.coords[0]) + "," + str(self.coords[1]) + "," + str(self.coords[2]) + "," + str(self.aveE)

	def printCSVLine(self):
		print(self.getCSVLine())
