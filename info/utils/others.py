"""
A set of utility functions.

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

corrcoefs()
read_SFPmatfile()
convert_matlabdatenum()

"""

import numpy as np
import scipy.io as sio
import datetime as dt

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


def read_SFPmatfile():
    """Reading SFP2_AllData.mat file."""
    matfile = sio.loadmat('SFP2_AllData.mat')

    data = matfile['OrigData']

    # Get DOY
    doy = data['data_raw_DOY'][0, 0]

    # Get the headers
    header = data['data_raw_header'][0, 0]

    # Get the raw data
    d = data['data_raw'][0, 0]

    return doy, header, d


def convert_matlabdatenum(matlab_datenum):
    """Convert the MATLAB datetime into Python datetime.

    Ref: http://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    """
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)

    return day + dayfrac
