"""
BCICIV2b.py

Description
-----------
This code is used to load EEG data from the BCICIV2b dataset.
It modifies the data to fit the requirements of the eegdata class,
which is used to store and process EEG data.

Dependencies
------------
numpy
pandas
scipy
mne

"""

import numpy as np
import scipy


def bciciv2b(
    subject: int = 1,
    session_list: list = None,
    run_list: list = None,
    labels=["left-hand", "right-hand"],
    path: str = "data/BCICIV2b/",
):
    """
    Description
    -----------

    Load EEG data from the BCICIV2b dataset.
    The data is loaded for a specific subject, session, and run.
    The data is filtered based on the event codes specified in the 'labels_dict'.

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list, optional
            list of session codes
        run_list : list, optional
            list of run numbers
        events_dict : dict
            dictionary mapping event names to event codes
        verbose : str
            verbosity level


    Returns:
    ----------
        eegdata: An instance of the eegdata class containing the loaded EEG data.

    """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != "/":
        path += "/"

    sfreq = 250.0
    events = {
        "get_start": [0, 3],
        "beep_sound": [2],
        "cue": [3, 4],
        "task_exec": [4, 7],
        "break": [7, 8.5],
    }
    ch_names = ["C3", "Cz", "C4"]
    ch_names = np.array(ch_names)
    tmin = 0.0

    if session_list is None:
        session_list = ["01T", "02T", "03T", "04E", "05E"]

    rawData, rawLabels = [], []

    for sec in session_list:
        file_name = "parsed_P%02d%s.mat" % (subject, sec)
        try:
            raw = scipy.io.loadmat(path + file_name)
        except:
            raise ValueError(
                "The file %s does not exist in the path %s" % (file_name, path)
            )

        rawData_ = raw["RawEEGData"]
        rawLabels_ = np.reshape(raw["Labels"], -1)
        rawData_ = np.reshape(
            rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2])
        )
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)

    X, y = np.concatenate(rawData), np.concatenate(rawLabels)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    labels_dict = {1: "left-hand", 2: "right-hand"}
    y = np.array([labels_dict[i] for i in y])
    selected_labels = np.isin(y, labels)
    X, y = X[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])

    return {
        "X": X,
        "y": y,
        "sfreq": sfreq,
        "y_dict": labels_dict,
        "events": events,
        "ch_names": ch_names,
        "tmin": tmin,
    }
