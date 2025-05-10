import numpy as np
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from methods.features.logpower import logpower  # função correta


def run_csp_logpower(subject_id, data_path="dataset/wcci2020/"):
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"].squeeze(1)  # [n_trials, channels, samples]
    y = np.array(dataset["y"])
    y = y + 1  # Corrige os rótulos de [0,1] para [1,2]

    X_band = np.expand_dims(X, axis=1)  # [n_trials, 1, channels, samples]
    transformer = csp()
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})["X"][
        :, 0
    ]  # [n_trials, components, samples]

    # Recoloca banda para reutilizar a função logpower (espera [n_trials, bands, channels, samples])
    X_reformatted = np.expand_dims(X_csp, axis=1)
    X_log = logpower({"X": X_reformatted}, flating=True)[
        "X"
    ]  # resultado: [n_trials, n_components]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_log, y)):
        X_train, X_test = X_log[train_idx], X_log[test_idx]
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
