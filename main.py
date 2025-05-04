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
from higuchi import HiguchiFractalEvolution
from logpower import LogPowerEnhanced

# Configura o logging (para exibir progresso e erros no terminal)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Cria diretórios de resultados se ainda não existirem
os.makedirs('results/Higuchi', exist_ok=True)
os.makedirs('results/LogPower', exist_ok=True)


# ======================= Classe Principal de Processamento =======================
class EEGProcessor:
    """
    Responsável por carregar os dados dos sujeitos e executar os experimentos
    com diferentes extratores de características.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_subject_data(self, subject_id, data_type='T'):
        """
        Carrega os dados .mat de um sujeito específico.
        Espera arquivos no formato: parsed_P01T.mat, parsed_P02T.mat, etc.
        """
        filename = f"parsed_P{subject_id:02d}{data_type}.mat"
        filepath = os.path.join(self.data_dir, filename)
        try:
            mat = scipy.io.loadmat(filepath)
            return mat['RawEEGData'], mat['Labels'].flatten()
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            return None, None

    def run_experiment(self, method_name, extractor, subject_ids):
        """
        Executa o experimento de classificação para cada sujeito, utilizando
        validação cruzada estratificada 5-fold.
        """
        accuracies = []  # Lista para armazenar as acurácias de cada sujeito

        for subject_id in tqdm(subject_ids, desc=f"Running {method_name}"):
            data, labels = self.load_subject_data(subject_id)
            if data is None:
                accuracies.append(np.nan)
                continue

            # Extrai as features com o método especificado (Higuchi ou LogPower)
            X = extractor.extract(data)

            # Cria 5 folds estratificados
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_acc = []

            for train_idx, test_idx in skf.split(X, labels):
                # Separa os dados em treino e teste
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Normaliza os dados
                scaler = StandardScaler()
                # Reduz a dimensão mantendo 95% da variância
                pca = PCA(n_components=0.95)

                X_train = scaler.fit_transform(X_train)
                X_train = pca.fit_transform(X_train)

                X_test = scaler.transform(X_test)
                X_test = pca.transform(X_test)

                # Seleciona as 30 melhores features (ou menos, dependendo da dimensionalidade pós-PCA)
                n_features_post_pca = X_train.shape[1]
                k = min(30, n_features_post_pca)
                if k < n_features_post_pca:
                    selector = SelectKBest(f_classif, k=k)
                    X_train = selector.fit_transform(X_train, y_train)
                    X_test = selector.transform(X_test)

                # Classificador LDA com shrinkage para melhor generalização
                clf = LDA(solver='lsqr', shrinkage='auto')
                clf.fit(X_train, y_train)
                fold_acc.append(clf.score(X_test, y_test))  # Acurácia do fold

            # Média das acurácias dos 5 folds
            acc = np.mean(fold_acc)
            accuracies.append(acc)

            # Mostra o resultado individual
            logging.info(
                f"{method_name} - P{subject_id:02d}: Accuracy = {acc:.4f}")

        return accuracies


# ======================= Função Principal =======================
def main():
    # Caminho da pasta com os dados de EEG
    DATA_DIR = "data/wcci2020/"
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return

    # Inicializa o processador com o caminho para os dados
    processor = EEGProcessor(DATA_DIR)

    # Define os sujeitos que serão avaliados (P01 a P10)
    subject_ids = range(1, 11)

    # Instancia os extratores de características
    higuchi = HiguchiFractalEvolution(kmax=10)
    logpower = LogPowerEnhanced()

    # Executa os experimentos com cada método
    higuchi_acc = processor.run_experiment("HFE", higuchi, subject_ids)
    logpower_acc = processor.run_experiment("LogPower", logpower, subject_ids)

    # Monta um relatório com os resultados
    report = pd.DataFrame({
        'Subject': [f'P{i:02d}' for i in subject_ids],
        'HFE': higuchi_acc,
        'LogPower': logpower_acc
    })

    print("\n=== Accuracy Report ===")
    print(report)
    print(f"\nHFE Mean: {np.nanmean(higuchi_acc):.4f}")
    print(f"LogPower Mean: {np.nanmean(logpower_acc):.4f}")

    # Teste estatístico de Wilcoxon para comparar os dois métodos
    valid_pairs = [(h, l) for h, l in zip(higuchi_acc, logpower_acc)
                   if not np.isnan(h) and not np.isnan(l)]
    if len(valid_pairs) >= 2:
        hig_valid, log_valid = zip(*valid_pairs)
        stat, p = wilcoxon(hig_valid, log_valid)
        print("\n=== Wilcoxon Test ===")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p:.4f}")
        if p < 0.05:
            print("Conclusion: Significant difference between methods (p < 0.05)")
            print("HFE is significantly better!" if np.mean(hig_valid) >
                  np.mean(log_valid) else "LogPower is significantly better!")
        else:
            print("Conclusion: No significant difference between methods")


if __name__ == "__main__":
    main()
