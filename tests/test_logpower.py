import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import LogPower


def test_logpower_init():
    """Testa a inicialização da classe LogPower com parâmetros padrão e personalizados."""
    # Testa inicialização com parâmetros padrão
    lp = LogPower()
    assert len(lp.freq_bands) == 5, f"Número incorreto de bandas padrão: {len(lp.freq_bands)}"
    assert lp.sfreq == 512, f"Valor padrão incorreto para sfreq: {lp.sfreq}"
    
    # Testa inicialização com parâmetros personalizados
    custom_bands = [
        ("alpha", 8, 13),
        ("beta", 13, 30),
    ]
    lp_custom = LogPower(freq_bands=custom_bands, sfreq=250)
    assert len(lp_custom.freq_bands) == 2, f"Número incorreto de bandas personalizadas: {len(lp_custom.freq_bands)}"
    assert lp_custom.sfreq == 250, f"Valor personalizado incorreto para sfreq: {lp_custom.sfreq}"


def test_logpower_extract():
    """Testa a extração de características de potência logarítmica de sinais EEG simulados."""
    lp = LogPower()
    
    # Cria dados de EEG simulados (3 trials, 4 canais, 512 amostras)
    n_trials, n_channels, n_samples = 3, 4, 512
    X = np.random.randn(n_trials, n_channels, n_samples)
    
    # Extrai características
    features = lp.extract(X)
    
    # Verifica dimensões corretas (n_trials x (n_channels * n_bands))
    expected_features = n_trials, n_channels * len(lp.freq_bands)
    assert features.shape == expected_features, f"Shape incorreto: {features.shape}, esperado: {expected_features}"
    
    # Verifica se não há valores NaN
    assert not np.isnan(features).any(), "Características contêm valores NaN"
    
    # Verifica se os valores são números reais (podem ser negativos devido ao log)
    assert np.isreal(features).all(), "Características contêm valores não reais"


def test_logpower_sine_waves():
    """Testa o comportamento do LogPower em ondas senoidais de diferentes frequências."""
    lp = LogPower()
    
    # Cria senos com frequências que correspondem às bandas de frequência
    t = np.linspace(0, 1, 512)
    
    test_freqs = {
        "delta": 2,      # dentro da banda delta (0.5-4 Hz)
        "theta": 6,      # dentro da banda theta (4-8 Hz)
        "alpha": 10,     # dentro da banda alpha (8-13 Hz)
        "beta": 20,      # dentro da banda beta (13-30 Hz)
        "gamma": 35      # dentro da banda gamma (30-40 Hz)
    }
    
    # Cria um trial com 5 canais, cada um com um seno de frequência diferente
    X = np.zeros((1, 5, 512))
    for i, (band, freq) in enumerate(test_freqs.items()):
        X[0, i, :] = np.sin(2 * np.pi * freq * t)
    
    # Extrai características
    features = lp.extract(X)
    
    # Verifica se cada canal tem maior potência na sua banda correspondente
    for i, band_name in enumerate(test_freqs.keys()):
        # Encontra o índice da banda no vetor de características
        band_idx = [j for j, (name, _, _) in enumerate(lp.freq_bands) if name == band_name][0]
        
        # Para cada canal, verifica se a banda correspondente tem a maior potência
        channel_features = features[0, i*len(lp.freq_bands):(i+1)*len(lp.freq_bands)]
        max_band_idx = np.argmax(channel_features)
        
        assert max_band_idx == band_idx, f"Canal {i} (banda {band_name}): banda de máxima potência incorreta"


def test_logpower_real_data():
    """Testa o cálculo de características de potência logarítmica em dados reais de EEG."""
    try:
        # Carrega dados de um sujeito para teste
        subject_id = 1
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]  # [n_trials, 1, channels, samples]
        y = np.array(dataset["y"]) + 1
        
        # Filtra classes 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]
        
        # Aplica banco de filtros
        eegdata_dict = {"X": X[:, np.newaxis, :, :], "sfreq": 512}
        eegdata_dict = filterbank(eegdata_dict, kind_bp="chebyshevII")
        X_filtered = eegdata_dict["X"]
        
        n_trials, n_bands, n_chans, n_filters, n_samples = X_filtered.shape
        X_reshaped = X_filtered.transpose(0, 1, 3, 2, 4).reshape(
            n_trials, n_bands * n_filters * n_chans, n_samples
        )
        
        # Extrai características
        lp = LogPower(sfreq=512)
        features = lp.extract(X_reshaped)
        
        # Verifica dimensões corretas e ausência de NaNs
        assert features.shape[0] == X.shape[0], "Número incorreto de exemplos nas características"
        assert not np.isnan(features).any(), "Características contêm valores NaN"
        
        # Verifica classificação simples para validar a utilidade das características
        X_scaled = StandardScaler().fit_transform(features)
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        train_idx, test_idx = next(skf.split(X_scaled, y))
        
        clf = LDA()
        clf.fit(X_scaled[train_idx], y[train_idx])
        accuracy = clf.score(X_scaled[test_idx], y[test_idx])
        
        # Verificar se a acurácia está acima do nível de chance (50%)
        assert accuracy > 0.5, f"Acurácia abaixo do nível de chance: {accuracy}"
        print(f"Acurácia na classificação com características de potência log: {accuracy:.4f}")
        
    except Exception as e:
        assert False, f"Erro ao testar com dados reais: {str(e)}"


def run_logpower_test():
    """Executa o teste completo do método LogPower em todos os sujeitos."""
    all_rows = []
    
    for subject_id in tqdm(range(1, 10), desc="LogPower"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]  # [n_trials, 1, channels, samples]
        y = np.array(dataset["y"]) + 1
        
        # Filtra classes 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]
        
        # Aplica banco de filtros
        eegdata_dict = {"X": X[:, np.newaxis, :, :], "sfreq": 512}
        eegdata_dict = filterbank(eegdata_dict, kind_bp="chebyshevII")
        X_filtered = eegdata_dict["X"]
        
        n_trials, n_bands, n_chans, n_filters, n_samples = X_filtered.shape
        X_reshaped = X_filtered.transpose(0, 1, 3, 2, 4).reshape(
            n_trials, n_bands * n_filters * n_chans, n_samples
        )
        
        # Extrai características
        lp = LogPower(sfreq=512)
        features = lp.extract(X_reshaped)
        features = StandardScaler().fit_transform(features)
        
        # Validação cruzada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        subject_rows = []
        
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
                    "predicted": np.argmax(probs[i]) + 1,
                }
                subject_rows.append(row)
                all_rows.append(row)
                
        # Salva resultados por sujeito
        os.makedirs("results/LogPower/Training", exist_ok=True)
        pd.DataFrame(subject_rows).to_csv(
            f"results/LogPower/Training/P{subject_id:02d}.csv", index=False
        )
    
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # Executa os testes unitários
    test_logpower_init()
    test_logpower_extract()
    test_logpower_sine_waves()
    print("Todos os testes unitários passaram!")
    
    # Executa o teste com dados reais
    test_logpower_real_data()
    print("Teste com dados reais passou!")
    
    # Executa o teste completo e exibe resultados
    df = run_logpower_test()
    acc = (df["true_label"] == df["predicted"]).mean()
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())
    
    print(f"LogPower Accuracy: {acc:.4f} | Média Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rótulos: {counts}")
