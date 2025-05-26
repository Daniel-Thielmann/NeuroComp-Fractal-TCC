import numpy as np
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from methods.features.fractal import HiguchiFractalEvolution


def run_csp_fractal(subject_id, data_path="dataset/wcci2020/"):
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra classes 1 e 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    # Garante shape correto: [n_trials, n_channels, n_samples]
    if X.ndim == 4 and X.shape[1] == 1:
        X = X.squeeze(1)
    if X.ndim != 3:
        raise ValueError(
            f"Esperado X com 3 dimensões (trials, channels, samples), mas recebeu shape {X.shape}"
        )

    # Centraliza os dados (como feito no Fractal puro)
    X = X - np.mean(X, axis=2, keepdims=True)

    # Aplica CSP direto no sinal original (SEM filterbank)
    X_csp_in = X[:, np.newaxis, :, :]  # [n_trials, 1, n_channels, n_samples]
    transformer = csp()
    transformer.fit({"X": X_csp_in, "y": y})
    X_csp = transformer.transform({"X": X_csp_in})[
        "X"
    ]  # [n_trials, 1, n_components, n_samples]
    X_csp = X_csp[:, 0]  # Remove dimensão da banda

    # Extrai features fractais dos componentes CSP
    hfd = HiguchiFractalEvolution(kmax=100)
    features = []
    for trial in X_csp:
        # Para cada trial, extraimos os dois primeiros componentes
        comps = (
            trial[:4] if trial.shape[0] >= 4 else trial
        )  # Usando 4 componentes para mais info
        trial_feat = []
        for comp in comps:
            # Adicionando informacao de energia ao lado das features fractais
            energy = np.log(np.mean(comp**2) + 1e-10)
            # Extraindo features fractais do componente
            slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(comp)
            # Caracteristicas estatisticas adicionais
            sk = np.std(comp)
            # Concatenando todas as caracteristicas por componente
            trial_feat.extend([slope, mean_lk, std_lk, energy, sk])
        features.append(trial_feat)

    features = np.array(features)
    # Normalizacao e reducao de dimensionalidade
    features = StandardScaler().fit_transform(features)
    # Aplica PCA para reduzir dimensionalidade (15 componentes)
    features = PCA(n_components=min(15, features.shape[1])).fit_transform(features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LDA()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        for i, idx in enumerate(test_idx):
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y_test[i],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
            )
    return rows
