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

    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "POz",
    ]
    # O dataset BCICIV2b tem 22 canais, mas alguns arquivos podem ter 25 (com POz e outros).
    # Para garantir compatibilidade, seleciona apenas os 22 primeiros canais padrão.
    ch_names = np.array(ch_names[:22])
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
                continue
            raw = scipy.io.loadmat(path + file_name)
            data = raw["signals"]
            # Se o arquivo tiver mais de 22 canais, seleciona apenas os 22 primeiros
            if data.shape[0] > 22:
                data = data[:22, :]
            sfreq = (
                raw["sfreq"].item() if hasattr(raw["sfreq"], "item") else raw["sfreq"]
            )
            events = raw["events"]
            labels_raw = raw["labels"].flatten()
            unique_codes = np.unique(events[:, 2])
            event_mapping = {}
            if 4 in unique_codes and 5 in unique_codes:
                count_4 = np.sum(events[:, 2] == 4)
                count_5 = np.sum(events[:, 2] == 5)
                if count_4 > 10 and count_5 > 10:
                    event_mapping = {4: 1, 5: 2}
            if not event_mapping and 1 in unique_codes and 2 in unique_codes:
                count_1 = np.sum(events[:, 2] == 1)
                count_2 = np.sum(events[:, 2] == 2)
                if count_1 > 10 and count_2 > 10:
                    event_mapping = {1: 1, 2: 2}
            if not event_mapping and 10 in unique_codes and 11 in unique_codes:
                count_10 = np.sum(events[:, 2] == 10)
                count_11 = np.sum(events[:, 2] == 11)
                if count_10 > 10 and count_11 > 10:
                    event_mapping = {10: 1, 11: 2}
            if not event_mapping:
                continue
            trial_events = []
            trial_labels = []
            for i, event in enumerate(events):
                event_time, _, event_code = event
                if event_code in event_mapping:
                    trial_events.append(event_time)
                    trial_labels.append(event_mapping[event_code])
            if len(trial_events) == 0:
                continue
            trial_length = int(4 * sfreq)
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
            if len(trials) > 0:
                trials_array = np.stack(trials, axis=0)
                rawData.append(trials_array)
                rawLabels.extend(valid_labels)
        except Exception:
            continue

    # Verificar se temos dados para processar
    if not rawData:
        raise ValueError("No data was loaded from any file")

    # Concatenate all data
    X = np.concatenate(rawData, axis=0)
    y = np.array(rawLabels)

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
