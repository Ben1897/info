"""
Noise generator.

@Author: Peishi Jiang <Ben1897>
@Date:   2017-02-12T14:14:08-06:00
@Email:  shixijps@gmail.com
@Last modified by:   Ben1897
@Last modified time: 2017-02-15T15:53:57-06:00

"""

import numpy as np


class noise(object):

    allowedNoiseDists = ['normal', 'uniform']

    def __init__(self, noiseDist, noisePara):
        '''
        noiseDist: the type of the noise [string]
        noisePara: the parameters of the noise term [list]
                   1     - variance coefficient [double]
                   2 ... - other parameters [any]
        '''
        self.dist = noiseDist
        self.para = noisePara

        self.checkNoise()
        self.initGenerator()

    def checkNoise(self):
        '''
        Check whether the required noise type is within the existing noise
        types.
        Output: NoneType
        '''
        #
        if self.dist is None:
            return lambda x: 0
        elif self.dist not in self.allowedNoiseDists:
            raise Exception('Unknown noise dist %s' % self.dist)

    def initGenerator(self):
        '''
        Create a generator for generating noise
        Output: NoneType
        '''
        if self.dist == 'uniform':
            self.generator = self.createUniformGenerator()
        elif self.dist == 'normal':
            self.generator = self.createNormalGenerator()

    def createUniformGenerator(self):
        '''
        Create a generator based on uniform distribution
        Output: NoneType
        '''
        varcoeff  = self.para[0]
        low, high = self.para[1], self.para[2]

        return lambda shape=None: varcoeff*np.random.uniform(low, high, shape)

    def createNormalGenerator(self):
        '''
        Create a generator based on normal distribution
        Output: NoneType
        '''
        varcoeff   = self.para[0]
        loc, scale = self.para[1], self.para[2]

        return lambda shape=None: varcoeff*np.random.normal(loc, scale, shape)

if __name__ == '__main__':
    # uniform
    noiseDist1 = 'uniform'
    noisePara1 = [3, 0, 10]
    noise_instance1 = noise(noiseDist1, noisePara1)

    # normal
    noiseDist2 = 'normal'
    noisePara2 = [1, 0, 1]
    noise_instance2 = noise(noiseDist2, noisePara2)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.hist(noise_instance1.generator([1000]))
    plt.figure(2)
    plt.hist(noise_instance2.generator([1000]))

    plt.show()
