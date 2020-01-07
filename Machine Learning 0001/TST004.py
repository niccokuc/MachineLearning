import numpy as np
import matplotlib.pyplot as plt

def plot11():
    t = np.arange( 0, 200, 1)
    N = len( t )
    y = np.random.rand( N )
    x = np.arange( 1, N+1 )
    labels = [ "data"+str(k) for k in range(1, N+1) ]
    samples = [ '' ] * N
    for i in range( 0, N, N/5 ):
        samples[i] = labels[i]
    width = 1.0
    plt.plot( x, y )
    plt.ylabel( 'Intensity' )
    plt.xticks(x + width/2.0, samples )
    plt.show()

plot11()