import sys
import os
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Usando LDA para obter melhor acurácia
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from methods.features.fractal import HiguchiFractalEvolution


def run_csp_fractal_all():
    all_rows = []

    # Usando kmax=100 para extrair Fractal mais refinado
    hfd = HiguchiFractalEvolution(kmax=100)

    for subject_id in tqdm(range(1, 10), desc="CSP_Fractal"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"].squeeze(1)
        y = np.array(dataset["y"]) + 1

        # Foco apenas em labels 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        # Aplica CSP para extrair componentes espaciais
        X_band = np.expand_dims(X, axis=1)  # [n_trials, 1, channels, samples]
        transformer = csp()
        transformer.fit({"X": X_band, "y": y})
        X_csp = transformer.transform({"X": X_band})["X"][
            :, 0
        ]  # [n_trials, components, samples]

        # Calcula Higuchi Fractal para cada componente CSP
        features = []
        for trial in X_csp:
            trial_feat = []
            for comp in trial:
                comp = comp - np.mean(comp)  # baseline correction
                slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(comp)
                trial_feat.extend([slope, mean_lk, std_lk])
            features.append(trial_feat)

        # Padroniza as features
        X_feat = np.array(features)
        X_feat = StandardScaler().fit_transform(X_feat)

        # Classificação com LDA (melhor desempenho para CSP+Fractal)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LDA()
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)

            for i, idx in enumerate(test_idx):
                all_rows.append(
                    {
                        "subject_id": subject_id,
                        "fold": fold_idx,
                        "true_label": y_test[i],
                        "left_prob": probs[i][0],
                        "right_prob": probs[i][1],
                    }
                )

        # Salva CSV por sujeito
        df_sub = pd.DataFrame([r for r in all_rows if r["subject_id"] == subject_id])
        os.makedirs("results/CSP_Fractal/Training", exist_ok=True)
        df_sub.to_csv(
            f"results/CSP_Fractal/Training/P{subject_id:02d}.csv", index=False
        )

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = run_csp_fractal_all()
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    acc = (df["true_label"] == df["left_prob"].lt(0.5).astype(int) + 1).mean()
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())
    print(
        f"Acurácia: {acc:.4f} | Média Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rótulos: {counts}"
    )
