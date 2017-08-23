import numpy as np

def datafile_values(data_filepath):
    """
    Extract data from data file
    return m: number of columns
    return X: input values with added intercept terms
    return y: output values
    """
    data = np.loadtxt(data_filepath, delimiter=',')
    data = np.asmatrix(data)

    m = data.shape[1]
    y = data[:, m-1]
    X = data[:, 0:m-1]
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)

    return m, y, X

def paramfile_values(param_filepath):
    """
    Extract parameters from parameter file
    """
    parameters = np.loadtxt(param_filepath, delimiter=',')
    parameters = np.append(1, parameters)
    return parameters
