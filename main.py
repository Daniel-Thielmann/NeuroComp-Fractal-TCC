import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import LogPower
from methods.features.fractal import HiguchiFractalEvolution
from methods.pipelines.csp_fractal import run_csp_fractal
from methods.pipelines.csp_logpower import run_csp_logpower
from methods.pipelines.fbcsp_fractal import run_fbcsp_fractal
from methods.pipelines.fbcsp_logpower import run_fbcsp_logpower
import sys
from pathlib import Path

# Adicionando a pasta de scripts ao path para importação
sys.path.append(str(Path("graphics/scripts")))

os.makedirs("results/summaries", exist_ok=True)


def run_fractal():
    hfd = HiguchiFractalEvolution(kmax=100)  # kmax maior para mais detalhe
    # selected_channels = ["C3", "C4", "CP3", "CP4", "FC3", "FC4", "CPz", "FCz"]
    os.makedirs("results/Fractal/Training", exist_ok=True)
    os.makedirs("results/Fractal/Evaluate", exist_ok=True)

    for subject_id in tqdm(range(1, 10), desc="Fractal"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"].squeeze(1)
        y = np.array(dataset["y"]) + 1
        ch_names = dataset["ch_names"]
        # selected_indices = [i for i, ch in enumerate(ch_names) if ch in selected_channels]
        # X = X[mask][:, selected_indices, :]
        mask = (y == 1) | (y == 2)
        X = X[mask]  # usa todos os canais
        y = y[mask]

        # Centraliza o sinal por canal/trial
        X = X - np.mean(X, axis=2, keepdims=True)

        features = hfd.extract(X)
        features = StandardScaler().fit_transform(features)
        # Redução de dimensionalidade para 15 componentes (ajuste conforme necessário)
        features = PCA(n_components=min(15, features.shape[1])).fit_transform(features)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        training_rows, evaluate_rows = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
            clf = LDA()
            clf.fit(features[train_idx], y[train_idx])
            probs = clf.predict_proba(features[test_idx])
            for i, idx in enumerate(test_idx):
                row = {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
                (training_rows if fold_idx < 4 else evaluate_rows).append(row)

        pd.DataFrame(training_rows).to_csv(
            f"results/Fractal/Training/P{subject_id:02d}.csv", index=False
        )
        pd.DataFrame(evaluate_rows).to_csv(
            f"results/Fractal/Evaluate/P{subject_id:02d}.csv", index=False
        )


def run_logpower():
    os.makedirs("results/LogPower/Training", exist_ok=True)
    os.makedirs("results/LogPower/Evaluate", exist_ok=True)

    for subject_id in tqdm(range(1, 10), desc="LogPower"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]
        y = np.array(dataset["y"]) + 1

        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        eegdata_dict = {"X": X[:, np.newaxis, :, :], "sfreq": 512}
        eegdata_dict = filterbank(eegdata_dict, kind_bp="chebyshevII")
        if not isinstance(eegdata_dict, dict) or "X" not in eegdata_dict:
            raise TypeError(
                f"Retorno inesperado de filterbank: {type(eegdata_dict)} - {eegdata_dict}"
            )
        X_filtered = eegdata_dict["X"]

        if X_filtered.ndim != 5:
            raise ValueError(f"Shape inesperado após filterbank: {X_filtered.shape}")

        n_trials, n_bands, n_chans, n_filters, n_samples = X_filtered.shape
        X_reshaped = X_filtered.transpose(0, 1, 3, 2, 4).reshape(
            n_trials, n_bands * n_filters * n_chans, n_samples
        )

        extractor = LogPower(sfreq=512)
        X_feat = extractor.extract(X_reshaped)
        X_feat = StandardScaler().fit_transform(X_feat)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        training_rows, evaluate_rows = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            clf = LDA()
            clf.fit(X_feat[train_idx], y[train_idx])
            probs = clf.predict_proba(X_feat[test_idx])
            for i, idx in enumerate(test_idx):
                row = {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
                (training_rows if fold_idx < 4 else evaluate_rows).append(row)

        pd.DataFrame(training_rows).to_csv(
            f"results/LogPower/Training/P{subject_id:02d}.csv", index=False
        )
        pd.DataFrame(evaluate_rows).to_csv(
            f"results/LogPower/Evaluate/P{subject_id:02d}.csv", index=False
        )


def log_summary(method_name):
    folder = f"results/{method_name}/Training"
    files = sorted(os.listdir(folder))

    # Handle empty folder case
    if not files:
        return f"[{method_name}] Sem dados disponíveis"

    # Read and concatenate files safely
    dfs = [pd.read_csv(os.path.join(folder, f)) for f in files]
    if not dfs:
        return f"[{method_name}] Sem dados disponíveis"

    # Use empty DataFrame check before concatenation
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)

    df = df[df["true_label"].isin([1, 2])]
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    acc = (df["true_label"] == df["left_prob"].lt(0.5).astype(int) + 1).mean()
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())
    return f"[{method_name}] Acuracia: {acc:.4f} | Media Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rotulos: {counts}"


def build_final_csv_and_wilcoxon():
    def extract_correct_prob(row):
        return row["left_prob"] if row["true_label"] == 1 else row["right_prob"]

    def load_all(method):
        all_probs = []
        for subset in ["Training", "Evaluate"]:
            folder = f"results/{method}/{subset}"
            if not os.path.exists(folder):
                continue
            files = sorted(os.listdir(folder))
            if not files:
                continue
            for file in files:
                df = pd.read_csv(os.path.join(folder, file))
                df = df[df["true_label"].isin([1, 2])]
                all_probs.extend(df.apply(extract_correct_prob, axis=1))
        return all_probs

    comparisons = [
        ("Fractal", "LogPower"),
        ("CSP_Fractal", "CSP_LogPower"),
        ("FBCSP_Fractal", "FBCSP_LogPower"),
    ]

    for m1, m2 in comparisons:
        print(f"\n=== Wilcoxon Test ({m1} vs {m2}) ===")
        vals1 = load_all(m1)
        vals2 = load_all(m2)

        min_len = min(len(vals1), len(vals2))
        vals1, vals2 = vals1[:min_len], vals2[:min_len]

        df_comp = pd.DataFrame({m1: vals1, m2: vals2})
        df_comp.to_csv(f"results/summaries/{m1}_vs_{m2}_comparison.csv", index=False)

        stat, p = wilcoxon(df_comp[m1], df_comp[m2])
        print(f"Statistic: {stat:.4f}")
        print(f"P-value  : {p:.4e}")
        print(
            "Conclusao:",
            "Diferenca significativa" if p < 0.05 else "Nao ha diferenca significativa",
        )


def run_kappa_analysis():
    """
    Executa a análise de kappa para todos os sujeitos e métodos.
    Importa o script de cálculo do kappa e executa suas funções.
    """
    try:
        print("\n=== ANÁLISE DE KAPPA POR SUJEITO E MÉTODO ===")
        from graphics.scripts.calculate_kappa_table import calculate_kappa, methods

        # Calcular kappa para cada sujeito e método
        results = pd.DataFrame(columns=["Sujeito"] + methods)

        for subject_id in range(1, 10):  # Sujeitos 1-9
            row = {"Sujeito": f"P{subject_id:02d}"}

            for method in methods:
                kappa = calculate_kappa(subject_id, method)
                row[method] = kappa

            # Criar DataFrame com a linha completa para evitar warning de concatenação
            new_row = pd.DataFrame([row])
            if not new_row.isnull().all(axis=1).iloc[0]:
                # Verificar se o DataFrame results está vazio
                if results.empty:
                    results = new_row.copy()
                else:
                    # Criar uma cópia do resultado antes da concatenação
                    results_copy = results.copy()
                    # Concatenar com cópia para evitar warnings
                    results = pd.concat([results_copy, new_row], ignore_index=True)

        # Adicionar linha de média
        mean_row = {"Sujeito": "Média"}
        for method in methods:
            mean_row[method] = results[method].mean()

        # Criar DataFrame com a linha de média completa
        mean_df = pd.DataFrame([mean_row])
        # Criar uma cópia do resultado antes da concatenação
        results_copy = results.copy()
        # Concatenar com cópia para evitar warnings
        results = pd.concat([results_copy, mean_df], ignore_index=True)

        # Formatar e exibir a tabela
        formatted_results = results.copy()
        for col in methods:
            formatted_results[col] = formatted_results[col].apply(
                lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
            )

        from tabulate import tabulate

        print("\nValores de Kappa para cada sujeito e método:")
        print(
            tabulate(
                formatted_results, headers="keys", tablefmt="grid", showindex=False
            )
        )

        # Conclusão com os melhores métodos
        best_method = results.iloc[-1, 1:].idxmax()
        best_value = results.iloc[-1, 1:].max()
        print(
            f"\nCONCLUSÃO: O método com melhor desempenho geral foi '{best_method}' com um kappa médio de {best_value:.4f}"
        )

        # Análise por sujeito
        best_subjects = {}
        for method in methods:
            best_subject = results[:-1]["Sujeito"][results[:-1][method].idxmax()]
            best_subjects[method] = best_subject
            best_kappa = results[:-1][method].max()
            print(
                f"- {method}: Melhor desempenho com o sujeito {best_subject} (kappa = {best_kappa:.4f})"
            )

        # Análise por sujeito - qual método funciona melhor para cada sujeito
        print("\nMelhor método para cada sujeito:")
        for i in range(len(results) - 1):  # Excluindo a linha de média
            subject = results.iloc[i]["Sujeito"]
            best_method_for_subject = results.iloc[i, 1:].idxmax()
            best_value_for_subject = results.iloc[i, 1:].max()
            print(
                f"- {subject}: {best_method_for_subject} (kappa = {best_value_for_subject:.4f})"
            )

        # Salvar os resultados em um arquivo CSV
        results.to_csv("results/summaries/kappa_by_subject_method.csv", index=False)

        return results
    except Exception as e:
        print(f"Erro ao executar análise de kappa: {e}")
        return None


def main():
    print("Running Fractal...")
    run_fractal()

    print("Running LogPower...")
    run_logpower()

    for name, func in [
        ("CSP_Fractal", run_csp_fractal),
        ("CSP_LogPower", run_csp_logpower),
        ("FBCSP_Fractal", run_fbcsp_fractal),
        ("FBCSP_LogPower", run_fbcsp_logpower),
    ]:
        os.makedirs(f"results/{name}/Training", exist_ok=True)
        os.makedirs(f"results/{name}/Evaluate", exist_ok=True)

        for subject_id in tqdm(range(1, 10), desc=name):
            rows = func(subject_id)
            df = pd.DataFrame(rows)
            df[df["fold"] < 4].to_csv(
                f"results/{name}/Training/P{subject_id:02d}.csv", index=False
            )
            df[df["fold"] == 4].to_csv(
                f"results/{name}/Evaluate/P{subject_id:02d}.csv", index=False
            )

    print("\n=== RESUMO FINAL DOS METODOS ===")
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
    run_kappa_analysis()


if __name__ == "__main__":
    main()
