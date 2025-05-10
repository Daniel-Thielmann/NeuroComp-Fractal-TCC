import numpy as np


def logpower(eegdata: dict, flating: bool = False) -> dict:
    """
    Calcula o Log Power dos sinais EEG contidos em eegdata['X'].

    Parameters
    ----------
    eegdata : dict
        Deve conter a chave 'X' com shape [n_trials, bands, channels, samples]
    flating : bool
        Se True, retorna dados flattenados por trial.

    Returns
    -------
    dict com chave 'X' contendo os dados transformados
    """
    X = eegdata["X"].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = np.log(np.mean(X[signal_] ** 2))
        X_.append(filtered)

    X_ = np.array(X_)
    shape = eegdata["X"].shape
    if flating:
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))
    else:
        X_ = X_.reshape((shape[0], shape[1], np.prod(shape[2:-1])))

    eegdata["X"] = X_
    return eegdata
