import numpy as np
from scipy.io import loadmat
from typing import List, Optional, Dict, Any


def cbcic(
    subject: int = 1,
    session_list: Optional[List[str]] = None,
    labels: List[str] = ["left-hand", "right-hand"],
    path: str = "data/cbcic/",
) -> Dict[str, Any]:
    """
    This function loads EEG data for a specific subject and session from the cbcic dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.

    The dataset can be found at:
    https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow

    Parameters
    ----------
    subject : int, optional
        Index of the subject to retrieve the data from. Must be between 1 and 10.
        Default is 1.
    session_list : list, optional
        List of session codes to load. Valid options are 'T' (training) and 'E' (evaluation).
        If None, all sessions are loaded. Default is None.
    labels : list, optional
        List of labels to include in the dataset. Valid options are 'left-hand' and 'right-hand'.
        Default is ['left-hand', 'right-hand'].
    path : str, optional
        Path to the folder containing the dataset files. Default is 'data/cbcic/'.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        X: EEG data as a numpy array.
        y: Labels corresponding to the EEG data.
        sfreq: Sampling frequency of the EEG data.
        y_dict: Mapping of labels to integers.
        events: Dictionary describing event markers.
        ch_names: List of channel names.
        tmin: Start time of the EEG data.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid or if the specified file does not exist.

    Examples
    --------
    Load EEG data for subject 1, all sessions, and default labels:

    >>> from bciflow.datasets import cbcic
    >>> eeg_data = cbcic(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    """

    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be an int type value")
    if subject > 10 or subject < 1:
        raise ValueError("subject has to be between 1 and 10")

    # Check if the session_list input is valid
    if type(session_list) != list and session_list is not None:
        raise ValueError("session_list has to be a list or None")
    if session_list is not None:
        for i in session_list:
            if i not in ["T", "E"]:
                raise ValueError("session_list has to be a sublist of ['T', 'E']")
    else:
        session_list = ["T", "E"]

    # Check if the labels input is valid
    if type(labels) != list:
        raise ValueError("labels has to be a list type value")
    for i in labels:
        if i not in ["left-hand", "right-hand"]:
            raise ValueError(
                "labels has to be a sublist of ['left-hand', 'right-hand']"
            )

    # Check if the path input is valid
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != "/":
        path += "/"

    # Set basic parameters of the clinical BCI challenge dataset
    sfreq = 512.0
    events = {
        "get_start": [0, 3],
        "beep_sound": [2],
        "cue": [3, 8],
        "task_exec": [3, 8],
    }
    ch_names = np.array(
        ["F3", "FC3", "C3", "CP3", "P3", "FCz", "CPz", "F4", "FC4", "C4", "CP4", "P4"]
    )
    tmin = 0.0

    rawData, rawLabels = [], []

    for sec in session_list:
        file_name = "parsed_P%02d%s.mat" % (subject, sec)
        try:
            raw = loadmat(path + file_name)
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
        "y_dict": y_dict,
        "events": events,
        "ch_names": ch_names,
        "tmin": tmin,
    }
