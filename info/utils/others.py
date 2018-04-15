"""
A set of utility functions.

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

corrcoefs()
butter_fiter()
aggregate()
normalize()
interpolate_nan()
parse_SFP()
read_SFPmatfile()
convert_matlabdatenum()
reorganize_data()

"""

import numpy as np
import pandas as pd
import scipy.io as sio
import datetime as dt
from scipy.signal import butter, lfilter


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


def butter_filter(data, N, fs, fc, btype='high'):
    """Filter the data by using the Butterworth filter."""
    # Get the relative cutoff frequency
    wn = fc/(.5*fs)

    # Filter the data
    b, a = butter(N, wn, btype=btype)
    filtered = lfilter(b, a, data)

    return filtered

def aggregate(data, interval, method='accumulate'):
    """Aggregate the data given the interval and the aggregation method."""
    # Get the size of the aggregated data set
    sizeo = data.size
    sizet = sizeo / interval

    # Initialize the target array
    datat = np.zeros(sizet)

    # Interpolate the nan values if any
    data = interpolate_nan(data)

    # Aggregation
    for i in range(sizet):
        d_temp = data[i*interval:(i+1)*interval]
        if method == 'accumulate':
            datat[i] = d_temp.sum()
        elif method == 'average':
            datat[i] = d_temp.mean()

    return datat

def normalize(data):
    """Normalize the data by subtracting its mean and dividing it by its std for each dimension.
    data -- [numpy array with shape(npts, ndim)]
    """
    npts, ndim = data.shape

    mean, std = np.mean(data, axis=0), np.std(data, axis=0)

    return (data - mean) / std

def interpolate_nan(data):
    """Interpolate nan values by using numpy interp method."""
    nans = np.isnan(data)
    if nans.any():
        nanloc = np.argwhere(nans).flatten()
        othloc = np.argwhere(~nans).flatten()
        othval = data[othloc]
        nanval = np.interp(nanloc, xp=othloc, fp=othval)
        data[nans] = nanval

    return data


def parse_SFP(filepath='./data/SFP2_AllData.mat'):
    """Parse SFP2_AllData.mat file."""
    ###############################
    # Read SFP2_AllData.mat file. #
    ###############################
    matfile = sio.loadmat(filepath)

    data = matfile['OrigData']

    # Get DOY
    doy = data['data_raw_DOY'][0, 0]

    # Get the headers
    header = data['data_raw_header'][0, 0]

    # Get the raw data
    d = data['data_raw'][0, 0]

    # Get the dimensions of the data
    nsec, ndim = d.shape

    #####################################################
    # Convert the MATLAB datetime into Python datetime. #
    #####################################################
    secsmat = d[:, 0]
    secspy  = np.zeros(nsec, dtype='datetime64[us]')
    for i in range(nsec):
        matlab_datenum = secsmat[i]
        day       = dt.datetime.fromordinal(int(matlab_datenum))
        daysec    = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
        # Round the datetime to minute
        time = day + daysec
        if time.second == 0:
            time = dt.datetime(year=time.year, month=time.month, day=time.day,
                               hour=time.hour, minute=time.minute)
        elif time.second == 59:
            time = dt.datetime(year=time.year, month=time.month, day=time.day,
                               hour=time.hour, minute=time.minute) + \
                               dt.timedelta(minutes=1)
        else:
            raise Exception('Unsuitable second %d' % time.second)
        secspy[i] = time

    return secspy, d, header, doy


def read_SFPmatfile(filepath='./SFP2_AllData.mat'):
    """Reading SFP2_AllData.mat file."""
    matfile = sio.loadmat(filepath)

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


def reorganize_data(data, w):
    """Reorganize the data based on the lag conditions w.
    Input:
    data -- the original data [ndarray with shape(npts1, ndim)]
    w -- the lag conditions [list]
         e.g., [(0,-1), (1,-1), (2, -2)]
    Output:
    redata -- the reorganized data [ndarray with shape(npts2, len(w))]
    """
    npts1, ndim = data.shape

    varset = [varlag[0] for varlag in w]
    # lagset = [-varlag[1] for varlag in w]  # Reverse the sign because negative (-) means earlier and positive (+) means later in the original w
    lagset = [varlag[1] for varlag in w]

    # Move the lagset so that all are non-negative
    minlag = np.min(lagset)
    lagset = [lag - minlag for lag in lagset]

    maxlag = np.max(lagset)
    npts2 = npts1 - maxlag
    ndim2 = len(w)
    redata = np.zeros([npts2, ndim2])

    for i in range(ndim2):
        var = varset[i]
        lag = lagset[i]
        # print var, lag, -maxlag+lag
        if -maxlag+lag == 0:
            redata[:, i] = data[lag:, var]
        else:
            redata[:, i] = data[lag:-maxlag+lag, var]

    return redata


def dropna(data):
    """
    Exclude the datapoint at a time step if it contains at least one nan.
    Input:
    data -- the original data [ndarray with shape(npts1, ndim)]
    Output:
    redata -- the filtered data [ndarray with shape(npts2, len(w))]
    """
    # Assemble data into a pandas framework
    df = pd.DataFrame(data)

    # Drop nan
    dfnew = df.dropna(axis=0, how='any')

    return dfnew.values

if __name__ == '__main__':
    # Test reorganize_data
    data = np.array([[1,2,3,4,5,6,7,8,9],
                     [1,2,3,4,15,6,7,8,9]]).T
    w = [(0,-1), (0,-2), (1,0), (1,-3)]
    print reorganize_data(data, w)
