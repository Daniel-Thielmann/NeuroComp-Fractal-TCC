"""
Módulo para carregar o dataset CBCIC, adaptado da biblioteca bciflow para suportar 10 sujeitos.
"""

import numpy as np
from scipy.io import loadmat
import os


def cbcic_local(
    subject=1,
    session_list=None,
    labels=["left-hand", "right-hand"],
    path="dataset/wcci2020/",
):
    """
    This function loads EEG data for a specific subject and session from the CBCIC dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    in the entire package.

    Parameters
    ----------
    subject : int
        The subject ID, between 1 and 10.
    session_list : list
        List of sessions to load. Default is None, which loads all available sessions.
    labels : list
        List of labels to select. Default is ['left-hand', 'right-hand'].
    path : str
        Path to the CBCIC data folder.

    Returns
    -------
    dict
        A dictionary containing the EEG data in the format:
        {
            'X': np.ndarray, shape (n_trials, 1, n_channels, n_samples),
            'y': np.ndarray, shape (n_trials,),
            'ch_names': list,
            'sfreq': int,
            'task_id': str,
            'label_dict': dict
        }
    """

    # Validação de parâmetros
    if subject < 1 or subject > 10:
        raise ValueError("subject has to be between 1 and 10")

    # Inicialização de variáveis
    X, y = [], []
    train_file = os.path.join(path, f"parsed_P{subject:02d}T.mat")
    eval_file = os.path.join(path, f"parsed_P{subject:02d}E.mat")

    # Carregando os dados de treinamento
    if os.path.exists(train_file):
        train_data = loadmat(train_file)
        X.append(train_data["epo_a"])
        y.append(np.zeros(train_data["epo_a"].shape[0]))
        X.append(train_data["epo_b"])
        y.append(np.ones(train_data["epo_b"].shape[0]))

    # Carregando os dados de avaliação
    if os.path.exists(eval_file):
        eval_data = loadmat(eval_file)
        X.append(eval_data["epo_a"])
        y.append(np.zeros(eval_data["epo_a"].shape[0]))
        X.append(eval_data["epo_b"])
        y.append(np.ones(eval_data["epo_b"].shape[0]))

    # Consolidando os dados
    X = np.vstack(X)
    y = np.hstack(y)

    # Adicionando uma dimensão extra para o formato padrão do pacote
    X = X[:, np.newaxis, :, :]

    # Obtendo nomes dos canais a partir do arquivo .locs
    locs_file = os.path.join(path, "eeglab_chan12_mod.locs")
    ch_names = []
    if os.path.exists(locs_file):
        with open(locs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    ch_names.append(parts[3])

    # Criando o dicionário de rótulos
    label_dict = {0: "left-hand", 1: "right-hand"}

    # Retornando o dicionário no formato padrão
    return {
        "X": X,
        "y": y,
        "ch_names": ch_names,
        "sfreq": 512,  # Frequência de amostragem do dataset CBCIC
        "task_id": f"CBCIC_subject_{subject}",
        "label_dict": label_dict,
    }
