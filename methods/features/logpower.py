"""
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
"""

import numpy as np


def logpower(eegdata: dict, flating: bool = False) -> dict:
    """
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
    """
    X = eegdata["X"]
    # Compatível com shape [n_trials, n_channels, n_samples]
    if X.ndim == 4 and X.shape[1] == 1:
        X = np.squeeze(X, axis=1)
    if X.ndim != 3:
        raise ValueError(
            "Esperado shape [n_trials, n_channels, n_samples] para logpower."
        )
    n_trials, n_channels, n_samples = X.shape
    features = np.zeros((n_trials, n_channels))
    for i in range(n_trials):
        for j in range(n_channels):
            features[i, j] = np.log(np.mean(X[i, j, :] ** 2) + 1e-10)
    eegdata["X"] = features
    return eegdata
