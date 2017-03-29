"""
A set of utility functions.

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

"""

import numpy as np


def corrcoefs(x, y, lagset):
    """Calculate the correlation coefficients between x and y given different lags."""
    lagsize = lagset.size

    rhoset = np.zeros(lagsize)

    # Compute the correlation coefficient between x[t] and y[t+lag]
    for i in range(lagsize):
        lag = lagset[i]
        if lag > 0:
            rhoset[i] = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        elif lag < 0:
            rhoset[i] = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        else:
            rhoset[i] = np.corrcoef(x, y)[0, 1]

    return rhoset
