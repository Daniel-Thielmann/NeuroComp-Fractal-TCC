'''
Description
-----------
This module implements the Log Power feature extractor, which computes the logarithm of the 
power of EEG signals. This feature is commonly used in BCI applications to characterize 
the energy of brain activity in specific frequency bands.

This function computes the Log Power of the input EEG data. The Log Power is calculated 
as the logarithm of the mean squared amplitude of the signal. The result is stored in 
the dictionary under the key 'X'.

Function
------------
'''
import numpy as np

def logpower(eegdata: dict, flating: bool = False) -> dict:
    ''' 
    Parameters
    ----------
    eegdata : dict
        The input data, where the key 'X' holds the raw signal.
    flating : bool, optional
        If True, the output data is returned in a flattened format (default is False).

    Returns
    -------
    output : dict
        The transformed data, with the Log Power stored under the key 'X'.
    '''
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = np.log(np.mean(X[signal_]**2))
        X_.append(filtered)

    X_ = np.array(X_)
    shape = eegdata['X'].shape
    if flating:
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))
    else:
        X_ = X_.reshape((shape[0], shape[1], np.prod(shape[2:-1])))

    eegdata['X'] = X_

    return eegdata
