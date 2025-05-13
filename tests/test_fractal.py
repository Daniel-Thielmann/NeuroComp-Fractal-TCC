import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Usando QDA para obter a melhor acurácia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bciflow.datasets import cbcic
from methods.features.fractal import HiguchiFractalEvolution

# Canais disponíveis nos arquivos .mat
eeg_channels = [
    "F3",
    "FC3",
    "C3",
    "CP3",
    "P3",
    "FCz",
    "CPz",
    "F4",
    "FC4",
    "C4",
    "CP4",
    "P4",
]


def run_all():
    all_rows = []
    hfd = HiguchiFractalEvolution(kmax=100)

    # Seleção dos canais motores válidos entre os 12 disponíveis
    selected_channels = ["C3", "C4", "CP3", "CP4", "FC3", "FC4", "CPz", "FCz"]
    selected_indices = [
        i for i, ch in enumerate(eeg_channels) if ch in selected_channels
    ]

    for subject_id in tqdm(range(1, 10), desc="Fractal"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"].squeeze(1)
        y = np.array(dataset["y"]) + 1

        mask = (y == 1) | (y == 2)
        X = X[mask][:, selected_indices, :]
        y = y[mask]

        features = []
        for trial in X:
            trial_feat = []
            for comp in trial:
                comp = comp - np.mean(comp)  # baseline correction
                slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(comp)
                trial_feat.extend([slope, mean_lk, std_lk])
            features.append(trial_feat)

        X_feat = np.array(features)
        X_feat = StandardScaler().fit_transform(X_feat)
        X_feat = PCA(n_components=15).fit_transform(X_feat)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = QDA()  # Versão que funcionou melhor
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

        df_sub = pd.DataFrame([r for r in all_rows if r["subject_id"] == subject_id])
        os.makedirs("results/Fractal/Training", exist_ok=True)
        df_sub.to_csv(f"results/Fractal/Training/P{subject_id:02d}.csv", index=False)

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = run_all()
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    acc = (df["true_label"] == df["left_prob"].lt(0.5).astype(int) + 1).mean()
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())

    # Média Probabilidade Correta: média das probabilidades atribuídas à classe verdadeira pelo modelo, independentemente de acertar ou não.
    # A média dessas probabilidades mostra o quanto o modelo confia na classe certa, mesmo quando erra a classificação final.
    # Use a média da probabilidade correta para saber o quão confiante e calibrado está o modelo, mesmo nos erros.

    print(
        f"Acurácia: {acc:.4f} | Média Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rótulos: {counts}"
    )
