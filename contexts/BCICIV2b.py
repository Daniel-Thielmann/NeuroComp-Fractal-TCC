"""
BCICIV2b.py

Description
-----------
This code is used to load EEG data from the BCICIV2b dataset. It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data.

Dependencies
------------
numpy
pandas
scipy

"""

import numpy as np
import pandas as pd
import scipy.io
import os


def bciciv2b(
    subject: int = 1,
    session_list: list = None,
    run_list: list = None,
    labels: list = ["left-hand", "right-hand"],
    path: str = "dataset/BCICIV2b/",
):
    """
    Description
    -----------

    Load EEG data from the BCICIV2b dataset.
    It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data.

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list, optional
            list of session codes
        run_list : list, optional
            list of run numbers
        labels : list
            list mapping event names to event codes
        path : str
            path to the dataset


    Returns:
    ----------
        eegdata: An instance of the eegdata class containing the loaded EEG data.

    """
    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")
    if type(labels) != list:
        raise ValueError("Has to be an List type")
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
    tmin = 0.0  # 'sfreq' is set to 250. This represents the sampling frequency of the EEG data.
    # 'events' is a dictionary that maps event names to their corresponding time intervals.
    # 'ch_names' is a list of channel names.
    # 'tmin' is set to 0, representing the starting time of the EEG data.

    if session_list is None:
        session_list = ["01T", "02T", "03T", "04E", "05E"]

    rawData, rawLabels = [], []
    # If 'session_list' is not provided, it is set to default sessions.
    # 'rawData' and 'rawLabels' are empty lists that will store the EEG data and labels for each session.

    for sec in session_list:
        file_name = "B%02d%s.mat" % (subject, sec)
        try:
            if not os.path.exists(path + file_name):
                print(f"Warning: File {file_name} does not exist. Skipping.")
                continue

            # Load MAT file using scipy
            raw = scipy.io.loadmat(path + file_name)

            # Debug: verificar chaves disponíveis
            available_keys = [k for k in raw.keys() if not k.startswith("__")]
            print(f"Chaves disponíveis em {file_name}: {available_keys}")

            # Get the data and sampling frequency
            data = raw["signals"]  # Shape: (channels, samples)
            sfreq = (
                raw["sfreq"].item() if hasattr(raw["sfreq"], "item") else raw["sfreq"]
            )
            events = raw["events"]
            labels_raw = raw["labels"].flatten()

            print(f"Loaded {file_name}: data shape={data.shape}, sfreq={sfreq}")
            print(
                f"Events shape: {events.shape}, Labels: {np.unique(labels_raw)}"
            )  # Extract trials based on events
            # BCICIV2b uses different event codes depending on the file:
            # Some files use codes 4,5 and others use codes 10,11
            # Both represent left-hand and right-hand motor imagery

            # Check which event codes are present
            unique_codes = np.unique(events[:, 2])

            # Event mapping for BCICIV2b (multiple possible mappings)
            event_mapping = {}

            # Pattern 1: codes 4,5 (60 occurrences each)
            if 4 in unique_codes and 5 in unique_codes:
                event_mapping = {
                    4: 1,  # left-hand
                    5: 2,  # right-hand
                }
            # Pattern 2: codes 10,11 (60 occurrences each)
            elif 10 in unique_codes and 11 in unique_codes:
                event_mapping = {
                    10: 1,  # left-hand
                    11: 2,  # right-hand
                }
            else:
                print(
                    f"Warning: Unknown event pattern in {file_name}. Unique codes: {unique_codes}"
                )
                continue

            # Find trial start events
            trial_events = []
            trial_labels = []

            for i, event in enumerate(events):
                event_time, _, event_code = event
                if event_code in event_mapping:
                    trial_events.append(event_time)
                    trial_labels.append(event_mapping[event_code])

            print(f"Found {len(trial_events)} trials in {file_name}")

            if len(trial_events) == 0:
                print(f"No valid trials found in {file_name}")
                continue

            # Extract trial data
            # For BCICIV2b, we extract 4 seconds after each cue (1000 samples at 250Hz)
            trial_length = int(4 * sfreq)  # 4 seconds
            n_channels = data.shape[0]
            trials = []
            valid_labels = []

            for i, (event_time, label) in enumerate(zip(trial_events, trial_labels)):
                start_sample = int(event_time)
                end_sample = start_sample + trial_length

                if end_sample <= data.shape[1]:
                    trial_data = data[:, start_sample:end_sample]
                    trials.append(trial_data)
                    valid_labels.append(label)
                else:
                    print(f"Trial {i} extends beyond data length. Skipping.")

            if len(trials) > 0:
                # Convert to numpy array (trials, channels, samples)
                trials_array = np.stack(trials, axis=0)
                rawData.append(trials_array)
                rawLabels.extend(valid_labels)
                print(
                    f"Extracted {len(trials)} trials from {file_name}: {trials_array.shape}"
                )

        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

    # Verificar se temos dados para processar
    if not rawData:
        raise ValueError("No data was loaded from any file")

    # Concatenate all data
    X = np.concatenate(rawData, axis=0)
    y = np.array(rawLabels)

    print(f"Final concatenated data: X={X.shape}, y={y.shape}")
    print(f"Unique labels: {np.unique(y)}")

    # Adicionar dimensão extra se necessário para compatibilidade
    if len(X.shape) == 3:
        X = X[:, np.newaxis, :, :]  # (trials, 1, channels, samples)

    # Mapear labels para strings
    labels_dict = {1: "left-hand", 2: "right-hand"}

    # Filtrar apenas labels válidos (que existem no dicionário)
    valid_labels_mask = np.isin(y, list(labels_dict.keys()))
    if not np.any(valid_labels_mask):
        raise ValueError(f"No valid labels found. Unique labels: {np.unique(y)}")

    X = X[valid_labels_mask]
    y = y[valid_labels_mask]

    # Converter labels numéricos para strings
    y_strings = np.array([labels_dict[i] for i in y])

    # Filtrar apenas os labels solicitados
    selected_labels = np.isin(y_strings, labels)
    X, y_strings = X[selected_labels], y_strings[selected_labels]

    # Criar dicionário de mapeamento e converter para índices
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y_strings])

    return {
        "X": X,
        "y": y,
        "sfreq": sfreq,
        "y_dict": y_dict,
        "events": events,
        "ch_names": ch_names,
        "tmin": tmin,
    }
