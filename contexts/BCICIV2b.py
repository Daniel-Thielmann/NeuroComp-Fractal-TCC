import numpy as np
import os
import scipy.io


# Classe compatível com bciflow para retorno dos dados
class EEGData:
    def __init__(self, X, y, sfreq, ch_names, y_dict):
        self.X = X  # (n_trials, n_channels, n_samples)
        self.y = y  # (n_trials,)
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.y_dict = y_dict


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


# Classe compatível com bciflow para retorno dos dados
class EEGData:
    def __init__(self, X, y, sfreq, ch_names, y_dict):
        self.X = X  # (n_trials, n_channels, n_samples)
        self.y = y  # (n_trials,)
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.y_dict = y_dict


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

    ch_names = ["EEG:C3", "EEG:Cz", "EEG:C4"]
    ch_names = np.array(ch_names)
    tmin = 0.0

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
            if data.shape[0] != 3:
                raise ValueError(
                    f"Arquivo {file_name} possui {data.shape[0]} canais, esperado 3 canais (EEG:C3, EEG:Cz, EEG:C4)."
                )
            sfreq = (
                raw["sfreq"].item() if hasattr(raw["sfreq"], "item") else raw["sfreq"]
            )
            events = raw["events"]
            labels_raw = raw["labels"].flatten()
            unique_codes = np.unique(events[:, 2])
            event_mapping = {}
            if 4 in unique_codes and 5 in unique_codes:
                event_mapping = {4: 1, 5: 2}
            elif 1 in unique_codes and 2 in unique_codes:
                event_mapping = {1: 1, 2: 2}
            elif 10 in unique_codes and 11 in unique_codes:
                event_mapping = {10: 1, 11: 2}
            else:
                continue
            trial_events = []
            trial_labels = []
            for i, event in enumerate(events):
                event_time, _, event_code = event
                if event_code in event_mapping and event_mapping[event_code] in [1, 2]:
                    trial_events.append(event_time)
                    trial_labels.append(event_mapping[event_code])
            if len(trial_events) == 0:
                continue
            trial_length = int(4 * sfreq)
            n_channels = data.shape[0]
            trials = []
            valid_labels = []
            for i, (event_time, label) in enumerate(zip(trial_events, trial_labels)):
                # Volta event_time e sfreq para float
                start_sample = int(event_time + 3 * sfreq)
                end_sample = start_sample + int(4 * sfreq)
                if end_sample <= data.shape[1]:
                    trial_data = data[:, start_sample:end_sample]
                    trials.append(trial_data)
                    valid_labels.append(label)
            if len(trials) > 0:
                trials_array = np.stack(trials, axis=0)
                rawData.append(trials_array)
                rawLabels.extend(valid_labels)
        except Exception as e:
            continue

    # Verificar se temos dados para processar
    if not rawData:
        raise ValueError("No data was loaded from any file")

    # Concatenate all data
    X = np.concatenate(rawData, axis=0)  # (n_trials, n_channels, n_samples)
    y = np.array(rawLabels)

    # Padronizar shape para [n_trials, 1, n_channels, n_samples]
    X = X[:, np.newaxis, :, :]  # Adiciona dimensão extra igual ao bciflow

    # Mapear labels para strings
    labels_dict = {1: "left-hand", 2: "right-hand"}
    y_strings = np.array([labels_dict[i] for i in y if i in labels_dict])

    # Filtrar apenas os labels solicitados
    selected_labels = np.isin(y_strings, labels)
    X = X[selected_labels]
    y_strings = y_strings[selected_labels]

    # Criar dicionário de mapeamento e converter para índices
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y_strings])

    return EEGData(X, y, int(round(sfreq)), ch_names, y_dict)
