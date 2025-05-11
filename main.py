# ===================== Configuração Inicial ====================== #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.decomposition import PCA
from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import logpower as logpower_fn
from methods.features.fractal import HiguchiFractalEvolution
from methods.pipelines.csp_fractal import run_csp_fractal
from methods.pipelines.csp_logpower import run_csp_logpower
from methods.pipelines.fbcsp_fractal import run_fbcsp_fractal
from methods.pipelines.fbcsp_logpower import run_fbcsp_logpower

os.makedirs("results/summaries", exist_ok=True)


# ================== Roda método Fractal (ex-Higuchi) ================== #
def run_fractal():
    all_rows = []
    hfd = HiguchiFractalEvolution(kmax=100)

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
        X_feat = PCA(n_components=min(15, X_feat.shape[1])).fit_transform(X_feat)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            clf = QDA()
            clf.fit(X_feat[train_idx], y[train_idx])
            probs = clf.predict_proba(X_feat[test_idx])

            for i, idx in enumerate(test_idx):
                all_rows.append(
                    {
                        "subject_id": subject_id,
                        "fold": fold_idx,
                        "true_label": y[idx],
                        "left_prob": probs[i][0],
                        "right_prob": probs[i][1],
                    }
                )

        df_sub = pd.DataFrame([r for r in all_rows if r["subject_id"] == subject_id])
        os.makedirs("results/Fractal/Training", exist_ok=True)
        df_sub.to_csv(f"results/Fractal/Training/P{subject_id:02d}.csv", index=False)

    return pd.DataFrame(all_rows)


# =================== LogPower =================== #
def run_logpower():
    all_rows = []

    for subject_id in tqdm(range(1, 10), desc="LogPower"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]
        y = np.array(dataset["y"]) + 1

        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        eegdata = {"X": X, "sfreq": 512}
        eegdata, _ = filterbank(eegdata, kind_bp="chebyshevII")
        X_feat = logpower_fn(eegdata, flating=True)["X"]
        X_feat = StandardScaler().fit_transform(X_feat)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            clf = LDA()
            clf.fit(X_feat[train_idx], y[train_idx])
            probs = clf.predict_proba(X_feat[test_idx])
            for i, idx in enumerate(test_idx):
                all_rows.append(
                    {
                        "subject_id": subject_id,
                        "fold": fold_idx,
                        "true_label": y[idx],
                        "left_prob": probs[i][0],
                        "right_prob": probs[i][1],
                    }
                )

        df_sub = pd.DataFrame([r for r in all_rows if r["subject_id"] == subject_id])
        os.makedirs("results/LogPower/Training", exist_ok=True)
        df_sub.to_csv(f"results/LogPower/Training/P{subject_id:02d}.csv", index=False)

    return pd.DataFrame(all_rows)


# =================== Log Summary =================== #
def log_summary(method_name):
    folder = f"results/{method_name}/Training"
    df = pd.concat(
        [pd.read_csv(os.path.join(folder, f)) for f in sorted(os.listdir(folder))]
    )
    df = df[df["true_label"].isin([1, 2])]
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    acc = (df["true_label"] == df["left_prob"].lt(0.5).astype(int) + 1).mean()
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())
    return f"[{method_name}] Acurácia: {acc:.4f} | Média Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rótulos: {counts}"


# =================== Wilcoxon =================== #
def build_final_csv_and_wilcoxon():
    def extract_correct_prob(row):
        return row["left_prob"] if row["true_label"] == 1 else row["right_prob"]

    def load_all(method):
        folder = f"results/{method}/Training"
        all_probs = []
        for file in sorted(os.listdir(folder)):
            df = pd.read_csv(os.path.join(folder, file))
            df = df[df["true_label"].isin([1, 2])]
            all_probs.extend(df.apply(extract_correct_prob, axis=1))
        return all_probs

    fractal_vals = load_all("Fractal")
    logpower_vals = load_all("LogPower")

    df_final = pd.DataFrame({"Fractal": fractal_vals, "LogPower": logpower_vals})
    df_final.to_csv("results/summaries/fractal_vs_logpower_comparison.csv", index=False)

    stat, p = wilcoxon(df_final["Fractal"], df_final["LogPower"])
    print("\n=== Wilcoxon Test (Fractal vs LogPower) ===")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value : {p:.4f}")
    if p < 0.05:
        print("Conclusão: Diferença significativa entre os métodos")
    else:
        print("Conclusão: Não há diferença significativa entre os métodos")


# =================== Execução =================== #
def main():
    print("Running Fractal...")
    run_fractal()

    print("Running LogPower...")
    run_logpower()

    print("Running CSP + Fractal...")
    for subject_id in tqdm(range(1, 10), desc="CSP + Fractal"):
        df = pd.DataFrame(run_csp_fractal(subject_id))
        os.makedirs("results/CSP_Fractal/Training", exist_ok=True)
        df.to_csv(f"results/CSP_Fractal/Training/P{subject_id:02d}.csv", index=False)

    print("Running CSP + LogPower...")
    for subject_id in tqdm(range(1, 10), desc="CSP + LogPower"):
        df = pd.DataFrame(run_csp_logpower(subject_id))
        os.makedirs("results/CSP_LogPower/Training", exist_ok=True)
        df.to_csv(f"results/CSP_LogPower/Training/P{subject_id:02d}.csv", index=False)

    print("Running FBCSP + Fractal...")
    for subject_id in tqdm(range(1, 10), desc="FBCSP + Fractal"):
        df = pd.DataFrame(run_fbcsp_fractal(subject_id))
        os.makedirs("results/FBCSP_Fractal/Training", exist_ok=True)
        df.to_csv(f"results/FBCSP_Fractal/Training/P{subject_id:02d}.csv", index=False)

    print("Running FBCSP + LogPower...")
    for subject_id in tqdm(range(1, 10), desc="FBCSP + LogPower"):
        df = pd.DataFrame(run_fbcsp_logpower(subject_id))
        os.makedirs("results/FBCSP_LogPower/Training", exist_ok=True)
        df.to_csv(f"results/FBCSP_LogPower/Training/P{subject_id:02d}.csv", index=False)

    print("\n=== RESUMO FINAL DOS MÉTODOS ===")
    for method in [
        "Fractal",
        "LogPower",
        "CSP_Fractal",
        "CSP_LogPower",
        "FBCSP_Fractal",
        "FBCSP_LogPower",
    ]:
        print(log_summary(method))

    build_final_csv_and_wilcoxon()


if __name__ == "__main__":
    main()
