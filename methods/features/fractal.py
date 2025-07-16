'''
Description
-----------
This module implements the Higuchi Fractal Dimension feature extractor, which computes the 
fractal dimension of EEG signals. This feature is useful for capturing the complexity and 
self-similarity of brain activity patterns, making it suitable for tasks like motor 
imagery classification.

The Higuchi Fractal Dimension is calculated using the Higuchi algorithm that estimates 
the fractal dimension of a time series.

Function
------------
'''
import numpy as np


def _calculate_higuchi_fd(signal, kmax=100):
    """
    Calculate Higuchi Fractal Dimension of a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    kmax : int
        Maximum k value for fractal calculation
        
    Returns
    -------
    float
        Higuchi Fractal Dimension
    """
    n = len(signal)
    if n < 10:
        return 0.0

    scales = np.unique(
        np.logspace(0, np.log10(min(kmax, n // 2)), num=10, dtype=int)
    )
    lk = np.zeros(len(scales))
    diff = np.abs(np.diff(signal))

    for i, k in enumerate(scales):
        sum_l, count = 0.0, 0
        for m in range(k):
            ix = np.arange(m, n, k)
            if len(ix) > 1:
                sum_l += np.sum(diff[ix[:-1]]) * (n - 1) / (len(ix) * k)
                count += 1
        lk[i] = np.log(sum_l / count) if count > 0 else 0.0

    valid = (lk != 0) & ~np.isinf(lk)
    if np.sum(valid) < 2:
        return 0.0

    slope = np.polyfit(np.log(1.0 / scales[valid]), lk[valid], 1)[0]
    return slope


def higuchi_fractal(eegdata: dict, flating: bool = False, kmax: int = 100) -> dict:
    """
    Compute the Higuchi Fractal Dimension of the input EEG data.
    
    Parameters
    ----------
    eegdata : dict
        The input data, where the key 'X' holds the raw signal.
    flating : bool, optional
        If True, the output data is returned in a flattened format (default is False).
    kmax : int, optional
        Maximum k value for fractal calculation (default is 100).

    Returns
    -------
    output : dict
        The transformed data, with the Higuchi Fractal Dimension stored under the key 'X'.
    """
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    for signal_ in range(X.shape[0]):
        # Remove mean to center the signal
        signal = X[signal_] - np.mean(X[signal_])
        hfd = _calculate_higuchi_fd(signal, kmax)
        X_.append(hfd)

    X_ = np.array(X_)
    shape = eegdata['X'].shape
    if flating:
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))
    else:
        X_ = X_.reshape((shape[0], shape[1], np.prod(shape[2:-1])))

    eegdata['X'] = X_

    return eegdata
