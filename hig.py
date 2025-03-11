import numpy as np


def hig(eegdata: dict) -> dict:

    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    X_ = []
    # substituir a logpower pelo higuchi e testar outras funções
    for signal_ in range(X.shape[0]):
        filtered = np.log(np.mean(X[signal_]**2))
        X_.append(filtered)

    X_ = np.array(X_)
    shape = eegdata['X'].shape
    X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))

    eegdata['X'] = X_

    print(eegdata)

    return eegdata
