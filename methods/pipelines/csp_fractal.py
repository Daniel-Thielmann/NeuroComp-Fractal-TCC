import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "contexts"))
from contexts.BCICIV2b import bciciv2b
from bciflow.modules.sf.csp import csp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from methods.features.fractal import higuchi_fractal


def run_csp_fractal(subject_id, data_path="dataset/BCICIV2b/"):
    """
    Executa o metodo CSP combinado com features fractais para classificacao de EEG.

    Args:
        subject_id: ID do sujeito a ser processado (1-9)
        data_path: Caminho para o diretorio com os dados

    Returns:
        Lista de dicionarios com os resultados de classificacao
    """
    dataset = bciciv2b(subject=subject_id, path=data_path)
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra classes 1 e 2 (BCICIV2b já retorna apenas left-hand e right-hand)
    # mask = (y == 1) | (y == 2)
    # X = X[mask]
    # y = y[mask]

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
    features = []
    for trial in X_csp:
        # Para cada trial, extraimos os componentes CSP
        comps = (
            trial[:4] if trial.shape[0] >= 4 else trial
        )  # Usando 4 componentes para mais info
        trial_feat = []
        for comp in comps:
            # Adicionando informacao de energia ao lado das features fractais
            energy = np.log(np.mean(comp**2) + 1e-10)
            
            # Extraindo dimensão fractal do componente usando nova função
            comp_data = {"X": comp.reshape(1, 1, -1)}  # [1 trial, 1 canal, samples]
            fractal_result = higuchi_fractal(comp_data, flating=True)
            fractal_dim = fractal_result["X"][0, 0]  # Extrai valor escalar
            
            # Caracteristicas estatisticas adicionais
            sk = np.std(comp)
            
            # Concatenando todas as caracteristicas por componente
            trial_feat.extend([fractal_dim, energy, sk])
        features.append(trial_feat)

    features = np.array(features)
    # Normalizacao e reducao de dimensionalidade
    features = StandardScaler().fit_transform(features)
    # Redução de dimensionalidade - ajustar componentes baseado no número de amostras e features
    n_components = min(15, features.shape[1], features.shape[0] - 1)
    if n_components > 0:
        features = PCA(n_components=n_components).fit_transform(features)
    else:
        print(f"Warning: Insufficient data for PCA. Using original features.")

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
