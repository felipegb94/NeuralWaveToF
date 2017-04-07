import numpy as np

def GenerateCirculantMatrix(f):
	N = f.size
	circulantMatrix = np.zeros((N,N))
	for i in range(0,N):
		circulantMatrix[i,:] = np.roll(f,i)
	return circulantMatrix

def Hamiltonian_f(x,shift):
	slope = 3/np.pi
	conds = [(x < shift),(x > shift) & (x < (np.pi/3)+shift), (x > (np.pi/3)+shift) & (x < np.pi + shift), (x > np.pi + shift) & (x < (4*np.pi/3) + shift), x > 2*np.pi + shift]
	funcs = [lambda x: 0, lambda x: slope*(x-shift), lambda x: 1,
             lambda x: -slope*(x) + (4+slope*shift), lambda x: 0]
	return np.piecewise(x, conds, funcs)