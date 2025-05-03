import numpy as np
from sklearn.preprocessing import StandardScaler


class LogPower:
    def __init__(self):
        pass

    def extract(self, eeg):
        X = []
        y = eeg.events["labels"]

        for trial in eeg.data:
            trial_features = [np.log(np.mean(np.square(channel)))
                              for channel in trial]
            X.append(trial_features)

        return StandardScaler().fit_transform(X), y
