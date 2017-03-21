# A class for generating values from a chaotic logistic network
#
# @Author: Peishi Jiang <Peishi>
# @Date:   2017-02-10T13:36:00-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-02-16T10:07:50-06:00

# Ref: Allison's TIPNets process network manuscript

from noise import noise
import numpy as np


class Logistic(object):

    allowedNoiseTypes = ['exclusive', 'additive', 'multiplicative']

    def __init__(self, n, adjM, lagM, noiseType='additive', noiseDist=None, noisePara=None):
        '''
        n:         the number of variables [int]
        adjM:      the adjacent matrix [numpy array]
        lagM:      the lag matrix [numpy array]
        noiseType: additive / multiplicative [string]
        noiseDist: the distribution of the noise [string]
        noisePara: the parameters of the noise term [list]
                   1     - variance coefficient [double]
                   2 ... - other parameters [any]
        '''
        self.n         = n
        self.adjM      = adjM
        self.lagM      = lagM
        self.checkMatrix()

        if noiseType not in self.allowedNoiseTypes:
            raise Exception('Unknown noise type %s!' % noiseType)
        else:
            self.noiseType = noiseType
            self.noise     = noise(noiseDist, noisePara)

        self.initLogistic()

    def checkMatrix(self):
        '''
        Check whether the dimensions of adjM and lagM are the same and complies
        with the variable number n.
        Output: NoneType
        '''
        nx_adj, ny_adj = self.adjM.shape
        nx_lag, ny_lag = self.lagM.shape
        # Check if adjM and lagM are a square matrix
        if nx_adj != ny_adj:
            raise Exception('adjM is not a square matrix!')
        if nx_lag != ny_lag:
            raise Exception('lagM is not a square matrix!')

        # Check if numbers of elements in each column of adjM and lagM equal to n
        if nx_adj != self.n:
            raise Exception('The # of elements in adjM does not equal to n!')
        if nx_lag != self.n:
            raise Exception('The # of elements in lagM does not equal to n!')

        return

    def initLogistic(self):
        '''
        Initialize the logistic equations.
        self.funcs: a list of functions with length self.n
        Output: NoneType.
        '''
        noiseGenerator = self.noise.generator

        def getFunctionForVar(w, i):
            k = np.nonzero(w)[0].size
            if k != 0:
                if self.noiseType == 'additive':
                    return lambda x: sum([w[i]*logisticEqn(x[i]) for i in range(self.n)]) / k + noiseGenerator()
                elif self.noiseType == 'exclusive':
                    return lambda x: sum([w[i]*logisticEqn(x[i]) for i in range(self.n)]) / k
                elif self.noiseType == 'multiplicative':
                    return lambda x: sum([w[i]*logisticEqn(x[i]) for i in range(self.n)]) / k + x[i]*noiseGenerator()
            else:
                if self.noiseType == 'additive' or self.noiseType == 'exclusive':
                    return lambda x: noiseGenerator()
                elif self.noiseType == 'multiplicative':
                    return lambda x: x*noiseGenerator()

        # Create a list of functions
        self.funcs = list(map(getFunctionForVar, self.adjM, range(self.n)))

    def simulate(self, nstep):
        '''
        Conduct the simulation given the number of steps nstep.
        Output: a numpy array with shape (n, nstep)
        '''
        maxLag = self.lagM.max()
        ntrash = max(1000, self.lagM.max())
        simul  = np.zeros([self.n, nstep+ntrash])
        noiseGenerator = self.noise.generator

        # Generate the initial values for the first ntrash terms using the noise
        simul[:, :ntrash] = noiseGenerator([self.n, ntrash])

        # Generate the remaining nstep values by using self.funcs
        for i in range(nstep):
            index = i+ntrash
            prex = simul[:, index-maxLag:index]
            x = self.getRequiredX(prex)
            simul[:, index] = map(lambda j: self.funcs[j](x[j]), range(self.n))
            # x1 = self.funcs[0](x[0])
            # x2 = self.funcs[1](x[1])
            # simul[:, index] = np.array([x1, x2])

        # Return the last nstep values
        return simul[:, ntrash:]

    def getRequiredX(self, prex):
        '''
        Get the required previous values based on lagM.
        Output: a numpy array with shape (n, n)
        '''
        x = np.zeros([self.n, self.n])
        for i in range(self.n):
            adj = -self.lagM[i]
            # x[i, :] = map(lambda j: prex[i, j] if j != 0 else 0., adj)
            x[i, :] = map(lambda j: prex[j, adj[j]] if adj[j] != 0 else 0., range(adj.size))
        return x


def logisticEqn(x):
    '''
    The logistic equation
    '''
    return 4*x*(1-x)


if __name__ == '__main__':
    # Parameters
    n    = 3
    lag  = 2
    adjM = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    lagM = np.array([[0, lag, 0], [lag, 0, lag], [lag, 0, 0]])
    # noise parameters
    noiseType = 'exclusive'
    noiseDist = 'uniform'
    noisePara = [1, 0, 1]
    nstep = 200
    # Initialize the logistic equations
    logistic = Logistic(n, adjM, lagM, noiseType, noiseDist, noisePara)
    # Simulate
    results = logistic.simulate(nstep)
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(range(nstep), results[0, :])
    plt.show()
