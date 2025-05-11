import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import logpower


def run_logpower_all():
    all_rows = []

    for subject_id in tqdm(range(1, 10), desc="Logpower"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]  # [n_trials, 1, channels, samples]
        y = np.array(dataset["y"]) + 1

        # Filtra labels 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        eegdata = filterbank(eegdata, kind_bp="chebyshevII")
        X_feat = logpower(eegdata, flating=True)["X"]
        X_feat = StandardScaler().fit_transform(X_feat)

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
        os.makedirs("results/LogPower/Training", exist_ok=True)
        df_sub.to_csv(f"results/LogPower/Training/P{subject_id:02d}.csv", index=False)

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = run_logpower_all()
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
