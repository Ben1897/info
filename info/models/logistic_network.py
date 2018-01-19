"""
A class for generating values from a chaotic logistic network.

@Author: Peishi Jiang <Peishi>
@Date:   2017-02-10T13:36:00-06:00
@Email:  shixijps@gmail.com
@Last modified by:   Ben1897
@Last modified time: 2017-02-16T10:07:50-06:00

Ref: Allison's TIPNets process network manuscript

"""

import numpy as np
from info.utils.noise import noise
# from ..utils.noise import noise


class Logistic(object):

    allowedNoiseTypes = ['additive', 'multiplicative']

    def __init__(self, n, adjM, lagM, e, ez, a=4., noiseType='additive',
                 noiseDist=None, noisePara=None, snrOn=False):
        """The initial function.

        n:         the number of variables [int]
        adjM:      the adjacent matrix [numpy array]
        lagM:      the lag matrix [numpy array]
        a:         the logistic equation coefficient [float]
        e:         the coupling strength with the histories of other nodes
        ez:        the coupling strength with the noise
        noiseType: exclusive/ additive / multiplicative [string]
        noiseDist: the distribution of the noise [string]
        noiseOn:   the 1/0 values for determining whether the noise is used for each variable [numpy array]
        noisePara: the parameters of the noise term [list]
                   1     - variance coefficient [double]
                   2 ... - other parameters [any]
        snrOn:     whether compute the signal-to-noise ratio [bool]

        """
        self.n         = n
        self.adjM      = adjM
        self.lagM      = lagM
        self.e         = e
        self.ez        = ez
        self.a         = a
        self.snrOn     = snrOn
        self.checkMatrix()

        # if noiseOn is None:
        #     self.noiseOn = np.zeros(n)
        # else:
        #     if len(noiseOn) != self.n:
        #         raise Exception('the size of noiseOn should be equal to %d!' % n)
        #
        #     noiseOnset = np.unique(noiseOn)
        #     if not np.in1d(noiseOnset, np.array([0, 1])).all():
        #         raise Exception('Values of noiseOn should be from {0, 1}')
        #     self.noiseOn = noiseOn

        if noiseType not in self.allowedNoiseTypes:
            raise Exception('Unknown noise type %s!' % noiseType)
        else:
            self.noiseType = noiseType
            self.noise     = noise(noiseDist, noisePara)

        self.initLogistic()

    def checkMatrix(self):
        """Check the dimensions of adjM and lagM.

        Check whether the dimensions of adjM and lagM are the same and complies
        with the variable number n.

        Output: NoneType

        """
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
        """Initialize the logistic equations.

        self.funcs: a list of functions with length self.n
        Output: NoneType.

        """
        noiseGenerator = self.noise.generator

        def getSignalFunction(w, i):
            e, ez, a = self.e, self.ez, self.a
            k = np.nonzero(w)[0].size
            if k != 0:
                return lambda x, i: (1-e)*logisticEqn(x[i],a) + (1-ez)*e*sum([w[j]*logisticEqn(x[j],a) for j in range(self.n)]) / k
            else:
                return lambda x, i: (1-e)*logisticEqn(x[i],a)

        def getNoiseFunction(w, i):
            e, ez, a = self.e, self.ez, self.a
            k = np.nonzero(w)[0].size
            if k != 0:
                if self.noiseType == 'additive':
                    return lambda x, i: e*ez*noiseGenerator()
                elif self.noiseType == 'multiplicative':
                    return lambda x, i: e*ez*x[i]*noiseGenerator()
            else:
                if self.noiseType == 'additive':
                    return lambda x, i: e*ez*noiseGenerator()
                elif self.noiseType == 'multiplicative':
                    return lambda x, i: e*ez*x*noiseGenerator()

        # Create a list of functions
        self.signalFuncs = list(map(getSignalFunction, self.adjM, range(self.n)))
        self.noiseFuncs  = list(map(getNoiseFunction, self.adjM, range(self.n)))

        # def getFunctionForVar(w, i):
        #     e, ez, a = self.e, self.ez, self.a
        #     k = np.nonzero(w)[0].size
        #     if k != 0:
        #         if self.noiseType == 'additive':
        #             return lambda x, i: (1-e)*logisticEqn(x[i],a) + (1-ez)*e*sum([w[j]*logisticEqn(x[j],a) for j in range(self.n)]) / k + e*ez*noiseGenerator()
        #         elif self.noiseType == 'multiplicative':
        #             return lambda x, i: (1-e)*logisticEqn(x[i],a) + (1-ez)*e*sum([w[j]*logisticEqn(x[j],a) for j in range(self.n)]) / k + e*ez*x[i]*noiseGenerator()
        #     else:
        #         if self.noiseType == 'additive':
        #             return lambda x, i: (1-e)*logisticEqn(x[i],a) + e*ez*noiseGenerator()
        #         elif self.noiseType == 'multiplicative':
        #             return lambda x, i: (1-e)*logisticEqn(x[i],a) + e*ez*x*noiseGenerator()

        # # Create a list of functions
        # self.funcs = list(map(getFunctionForVar, self.adjM, range(self.n)))

    def simulate(self, nstep):
        """
        Conduct the simulation given the number of steps nstep.

        Output: a numpy array with shape (n, nstep)

        """
        snrOn  = self.snrOn
        maxLag = self.lagM.max()
        ntrash = max(1000, self.lagM.max())
        simul  = np.zeros([self.n, nstep+ntrash])
        signal = np.zeros([self.n, nstep+ntrash])
        noise  = np.zeros([self.n, nstep+ntrash])
        noiseGenerator = self.noise.generator

        # Generate the initial values for the first ntrash terms using the noise
        simul[:, :ntrash] = noiseGenerator([self.n, ntrash])

        # Generate the remaining nstep values by using self.funcs
        for i in range(nstep):
            index = i+ntrash
            prex = simul[:, index-maxLag:index]
            # Get the required x values at the ith step for the computation at the i+1th step
            x = self.getRequiredX(prex)
            # Compute the signal values
            signal[:, index] = map(lambda j: self.signalFuncs[j](x[j], j), range(self.n))
            # Compute the noise values
            noise[:, index] = map(lambda j: self.noiseFuncs[j](x[j], j), range(self.n))
            # Get the simulated values
            # simul[:, index] = map(lambda j: self.funcs[j](x[j], j), range(self.n))
            simul[:, index] = signal[:, index] + noise[:, index]

        # Return the last nstep values
        if snrOn:
            snr = np.var(signal[:, ntrash:]) / np.var(noise[:, ntrash:])
            return simul[:, ntrash:], signal[:, ntrash:], noise[:, ntrash:], snr
        else:
            return simul[:, ntrash:]

    def getRequiredX(self, prex):
        """
        Get the required previous values based on lagM.

        Output: a numpy array with shape (n, n)

        """
        x = np.zeros([self.n, self.n])
        for i in range(self.n):
            adj = -self.lagM[i]
            x[i, :] = map(lambda j: prex[j, adj[j]] if adj[j] != 0 else 0., range(adj.size))
        return x


def logisticEqn(x, a):
    """The logistic equation."""
    # return 4*x*(1-x)
    return a*x*(1-x)


if __name__ == '__main__':
    # Parameters
    n    = 3
    lag  = 2
    e, ez, a = 1., .5, 4.
    adjM = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    lagM = np.array([[0, lag, 0], [lag, 0, lag], [lag, 0, 0]])
    snrOn = True
    # noise parameters
    noiseType = 'additive'
    noiseDist = 'uniform'
    noisePara = [1, 0, 1]
    nstep = 10
    # Initialize the logistic equations
    logistic = Logistic(n, adjM, lagM, e=e, ez=ez, a=a, snrOn=snrOn,
                        noiseType=noiseType, noiseDist=noiseDist, noisePara=noisePara)
    # Simulate
    if not snrOn:
        results = logistic.simulate(nstep)
    else:
        results, _, _, snr = logistic.simulate(nstep)
        print snr
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(range(nstep), results[0, :])
    plt.show()
