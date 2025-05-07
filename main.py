import os
import numpy as np
import scipy.io
import pandas as pd
import logging
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from methods.features.higuchi import HiguchiFractalEvolution
from methods.features.logpower import LogPowerEnhanced
from methods.pipelines.csp_fractal import run_csp_fractal


# ============================== Configuração Inicial ============================= #

# Configura o logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Cria diretórios de resultados se ainda não existirem
for method in ['Higuchi', 'LogPower']:
    for phase in ['Training', 'Evaluate']:
        os.makedirs(f'results/{method}/{phase}', exist_ok=True)


# ======================= Classe Principal de Processamento ======================= #
# ============================ Gera CSVs de "training" ============================ #

class EEGProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_subject_data(self, subject_id, data_type='T'):
        filename = f"parsed_P{subject_id:02d}{data_type}.mat"
        filepath = os.path.join(self.data_dir, filename)
        try:
            mat = scipy.io.loadmat(filepath)
            return mat['RawEEGData'], mat['Labels'].flatten()
        except Exception as e:
            logging.error(f"Erro ao carregar {filename}: {str(e)}")
            return None, None

    def run_experiment(self, method_name, extractor, subject_ids):
        for subject_id in tqdm(subject_ids, desc=f"Running {method_name}"):
            data, labels = self.load_subject_data(subject_id)
            if data is None:
                continue

            X = extractor.extract(data)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                clf = LDA()
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)

                for i, idx in enumerate(test_idx):
                    rows.append({
                        'subject_id': subject_id,
                        'fold': fold_idx,
                        'true_label': y_test[i],
                        'left_prob': probs[i][0],
                        'right_prob': probs[i][1],
                    })

            df = pd.DataFrame(rows)
            output_path = f'results/{method_name}/Training/P{subject_id:02d}.csv'
            df.to_csv(output_path, index=False)


# ==================== Usa a classe Principal de Processamento ==================== #
# ============================ Gera CSVs de "evaluate" ============================ #

def generate_evaluation_csvs(processor, extractor, method_name):
    output_dir = f"results/{method_name}/Evaluate"
    os.makedirs(output_dir, exist_ok=True)

    for subject_id in range(1, 11):
        data, labels = processor.load_subject_data(subject_id)
        if data is None:
            continue

        X = extractor.extract(data)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = LDA()
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)

            for i, idx in enumerate(test_idx):
                rows.append({
                    'subject_id': subject_id,
                    'fold': fold_idx,
                    'true_label': y_test[i],
                    'left_prob': probs[i][0],
                    'right_prob': probs[i][1],
                })

        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/P{subject_id:02d}.csv", index=False)



# =================== Unifica os 40 csvs gerando um csv final =================== #
# ======================== Aplica Wilcoxon no csv final ========================= #

def build_final_csv_and_wilcoxon():
    def extract_prob(row):
        if row['true_label'] == 1:
            return row['left_prob']
        elif row['true_label'] == 2:
            return row['right_prob']
        return None

    def load_all_probs(method):
        all_probs = []
        for phase in ['Training', 'Evaluate']:
            folder = f"results/{method}/{phase}"
            for file in sorted(os.listdir(folder)):
                df = pd.read_csv(os.path.join(folder, file))
                df = df[df['true_label'].isin([1, 2])]
                all_probs.extend(df.apply(extract_prob, axis=1))
        return all_probs

    higuchi_values = load_all_probs("Higuchi")
    logpower_values = load_all_probs("LogPower")

    df_final = pd.DataFrame({
        "Higuchi": higuchi_values,
        "LogPower": logpower_values
    })
    df_final.to_csv("results/summaries/higuchi_vs_logpower_comparison.csv", index=False)

    # Estatísticas descritivas antes do Wilcoxon
    higuchi_mean = df_final["Higuchi"].mean()
    higuchi_std = df_final["Higuchi"].std()
    logpower_mean = df_final["LogPower"].mean()
    logpower_std = df_final["LogPower"].std()

    print("\n=== Estatísticas descritivas ===")
    print(f"Higuchi  -> Média: {higuchi_mean:.4f} | Desvio Padrão: {higuchi_std:.4f}")
    print(f"LogPower -> Média: {logpower_mean:.4f} | Desvio Padrão: {logpower_std:.4f}")

    stat, p = wilcoxon(df_final["Higuchi"], df_final["LogPower"])
    print("\n=== Wilcoxon Test (40 CSVs combinados) ===")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value : {p:.4f}")
    if p < 0.05:
        print("Conclusão: Diferença significativa entre os métodos")
    else:
        print("Conclusão: Não há diferença significativa entre os métodos")


# =============================== Execução =============================== #
# ============================== do Pipeline ============================= #


def main():
    DATA_DIR = "dataset/wcci2020/"
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return

    processor = EEGProcessor(DATA_DIR)
    subject_ids = range(1, 11)
    higuchi = HiguchiFractalEvolution(kmax=10)
    logpower = LogPowerEnhanced()

    processor.run_experiment("Higuchi", higuchi, subject_ids)
    processor.run_experiment("LogPower", logpower, subject_ids)

    generate_evaluation_csvs(processor, higuchi, "Higuchi")
    generate_evaluation_csvs(processor, logpower, "LogPower")

 # ==================== Executa CSP + Fractal ==================== #
    for subject_id in tqdm(subject_ids, desc="Running CSP + Fractal"):
        rows = run_csp_fractal(subject_id)
        df = pd.DataFrame(rows)
        output_dir = "results/CSP_Fractal/Training"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}/P{subject_id:02d}.csv", index=False)

    build_final_csv_and_wilcoxon()

if __name__ == "__main__":
    main()
