import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def PlotN(x,Y,xlabel='x',ylabel='y',title='plot'):
	
	nLines = Y.shape[0]
	for i in range(0,nLines):
		plt.plot(x, Y[i,:], '-',linewidth=1.5)
		plt.title(title,fontsize=16,fontweight='bold',)
		plt.xlabel(xlabel,fontsize=14,fontweight='bold',)
		plt.ylabel(ylabel,fontsize=14,fontweight='bold',)

	# plt.ylim([1.2*np.min(Y),1.2*np.max(Y)])

	plt.show()