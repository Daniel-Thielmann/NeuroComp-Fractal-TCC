"""
BCICIV2a.py

Description
-----------
This code is used to load EEG data from the BCICIV2a dataset. It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data.

Dependencies
------------
numpy
pandas
scipy
mne

"""

import numpy as np
import pandas as pd
import scipy.io


def bciciv2a(
    subject: int = 1,
    session_list: list = None,
    run_list: list = None,
    labels: list = ["left-hand", "right-hand", "both-feet", "tongue"],
    path: str = "data/BCICIV2a/",
):
    """
    Description
    -----------

    Load EEG data from the BCICIV2a dataset.
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
        verbose : str
            verbosity level


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
        "get_start": [0, 2],
        "beep_sound": [0],
        "cue": [2, 3.25],
        "task_exec": [3, 6],
        "break": [6, 7.5],
    }
    ch_names = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]
    ch_names = np.array(ch_names)
    tmin = 0.0
    # 'sfreq' is set to 250. This represents the sampling frequency of the EEG data.
    # 'events' is a dictionary that maps event names to their corresponding time intervals.
    # 'ch_names' is a list of channel names.
    # 'tmin' is set to 0, representing the starting time of the EEG data.

    if session_list is None:
        session_list = ["T", "E"]

    rawData, rawLabels = [], []
    # If 'session_list' is not provided, it is set to ['T', 'E'].
    # 'rawData' and 'rawLabels' are empty lists that will store the EEG data and labels for each session.
    for sec in session_list:
        file_name = "parsed_P%02d%s.mat" % (subject, sec)
        try:
            raw = scipy.io.loadmat(path + file_name)
        except:
            raise ValueError(
                "The file %s does not exist in the path %s" % (file_name, path)
            )

        # Debug: verificar chaves disponíveis
        available_keys = [k for k in raw.keys() if not k.startswith("__")]
        # print removido

        # Tentar diferentes nomes possíveis para os dados EEG
        data_key = None
        possible_data_keys = [
            "RawEEGData",
            "X",
            "data",
            "eeg_data",
            "signals",
            "EEG",
            "trials",
        ]
        for key in possible_data_keys:
            if key in raw:
                data_key = key
                break

        if data_key is None:
            raise ValueError(
                f"Não foi possível encontrar dados EEG em {file_name}. Chaves disponíveis: {available_keys}"
            )

        rawData_ = raw[data_key]
        # print removido

        # Tentar diferentes nomes possíveis para os labels
        label_key = None
        possible_label_keys = ["Labels", "y", "labels", "classes", "targets"]
        for key in possible_label_keys:
            if key in raw:
                label_key = key
                break

        if label_key is None:
            raise ValueError(
                f"Não foi possível encontrar labels em {file_name}. Chaves disponíveis: {available_keys}"
            )

        rawLabels_ = np.reshape(raw[label_key], -1)

        # Verificar o formato dos dados e ajustar conforme necessário
        # print removido

        # Para dados contínuos do BCICIV2a, precisamos segmentar em trials
        # Vamos assumir que os dados são contínuos e precisam ser segmentados
        if len(rawData_.shape) == 2:
            # Dados contínuos (channels, samples)
            n_channels, n_samples = rawData_.shape

            # Buscar informação de eventos se disponível
            if "events" in raw:
                events = raw["events"]
                # print removido

            # Para BCICIV2a, vamos usar uma segmentação fixa baseada nos labels
            # Cada trial tem aproximadamente 4 segundos (1000 samples a 250Hz)
            sfreq = raw.get("sfreq", 250)
            if isinstance(sfreq, np.ndarray):
                sfreq = sfreq.item()

            trial_length = int(4 * sfreq)  # 4 segundos por trial
            n_trials = len(rawLabels_)

            # print removido

            # Verificar se temos amostras suficientes
            total_needed = n_trials * trial_length
            if n_samples < total_needed:
                # print removido
                # Ajustar o comprimento do trial
                trial_length = n_samples // n_trials
                # print removido

            # Segmentar os dados em trials
            trials = []
            for i in range(n_trials):
                start_idx = i * trial_length
                end_idx = start_idx + trial_length
                if end_idx <= n_samples:
                    trial = rawData_[:, start_idx:end_idx]
                    trials.append(trial)
                else:
                    # Para o último trial, usar o que sobrou
                    trial = rawData_[:, start_idx:n_samples]
                    # Fazer padding se necessário
                    if trial.shape[1] < trial_length:
                        padding = np.zeros((n_channels, trial_length - trial.shape[1]))
                        trial = np.concatenate([trial, padding], axis=1)
                    trials.append(trial)

            # Converter para array (trials, channels, samples)
            rawData_ = np.stack(trials, axis=0)
            # print removido

        elif len(rawData_.shape) == 3:
            # Já tem o formato correto (trials, channels, samples)
            pass
        else:
            raise ValueError(f"Formato inesperado dos dados: {rawData_.shape}")

        # print removido
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)

    """
    For each session in the 'session_list', the raw EEG data is loaded using mne.io.read_raw_gdf.
    The data is filtered to include only the first 22 channels.
    The 'annotations' (relevant timestamps) are extracted and converted to a DataFrame.
    The onset times are normalized to start from zero.
    The event descriptions are converted to integers.
    The 'new_trial_time' is obtained by extracting the onset times of the '768' event ('768' is the code for new trial in the dataset).
    The 'times_' array is obtained from the raw data.
    The EEG data is extracted for each trial based on the 'new_trial_time'.
    The data is reshaped to include only the relevant channels and time points.
    The class labels are loaded from the corresponding .mat file.    The raw data and labels are appended to the 'rawData' and 'rawLabels' lists.
    """

    # Verificar se temos dados para concatenar
    if not rawData:
        raise ValueError("Nenhum dado foi carregado")

    # Se temos apenas uma sessão, usar diretamente
    if len(rawData) == 1:
        X = rawData[0]
        y = rawLabels[0]
    else:
        # Para múltiplas sessões, encontrar o menor tamanho temporal comum
        min_samples = min(data.shape[-1] for data in rawData)
        # print removido

        # Cortar todos os dados para o mesmo tamanho
        rawData_trimmed = [data[:, :, :min_samples] for data in rawData]

        # Concatenar dados de todas as sessões
        X = np.concatenate(
            rawData_trimmed, axis=0
        )  # Concatenar ao longo da dimensão de trials
        y = np.concatenate(rawLabels)

    # print removido

    # Adicionar dimensão extra se necessário para compatibilidade
    if len(X.shape) == 3:
        X = X[:, np.newaxis, :, :]  # (trials, 1, channels, samples)

    # Mapear labels para strings
    labels_dict = {1: "left-hand", 2: "right-hand", 3: "both-feet", 4: "tongue"}

    # Filtrar apenas labels válidos (que existem no dicionário)
    valid_labels_mask = np.isin(y, list(labels_dict.keys()))
    if not np.any(valid_labels_mask):
        raise ValueError(
            f"Nenhum label válido encontrado. Labels únicos: {np.unique(y)}"
        )

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
