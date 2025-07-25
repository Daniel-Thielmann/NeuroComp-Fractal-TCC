import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "contexts"))
from contexts.BCICIV2b import bciciv2b
from bciflow.modules.tf.filterbank import filterbank
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.fs.mibif import MIBIF
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def run_fbcsp_logpower(subject_id: int, data_path="dataset/BCICIV2b/"):
    """
    Executa o metodo FBCSP (Filter Bank CSP) combinado com Log Power para classificacao de EEG.
    
    Pipeline: Filter Bank -> CSP -> LogPower -> MIBIF -> LDA

    Args:
        subject_id: ID do sujeito a ser processado (1-9)
        data_path: Caminho para o diretorio com os dados

    Returns:
        Lista de dicionarios com os resultados de classificacao
    """
    # Carrega os dados
    dataset = bciciv2b(subject=subject_id, path=data_path)
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # 1. Filter Bank (Temporal Filtering)
    eegdata = {"X": X, "sfreq": 250}  # BCICIV2b usa 250Hz
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")
    if not isinstance(eegdata, dict) or "X" not in eegdata:
        raise TypeError(
            f"Retorno inesperado de filterbank: {type(eegdata)} - {eegdata}"
        )
    
    # 2. CSP (Spatial Filtering)
    sf = csp()
    eegdata = sf.fit_transform(eegdata)
    
    # 3. LogPower (Feature Extraction)
    eegdata = logpower(eegdata, flating=True)
    
    # Validacao cruzada com 5 folds
    results = []
    X_feat = eegdata["X"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
        # Preparar dados de treino e teste
        eegdata_train = {"X": X_feat[train_idx], "y": y[train_idx]}
        eegdata_test = {"X": X_feat[test_idx]}
        
        # 4. MIBIF (Feature Selection) - apenas no conjunto de treino
        fs = MIBIF(n_features=8, clf=LDA(), paired=False)
        fs.fit(eegdata_train)
        
        # Aplicar seleção de características em treino e teste
        X_train_selected = fs.transform(eegdata_train)["X"]
        X_test_selected = fs.transform(eegdata_test)["X"]
        
        # 5. LDA (Classification)
        clf = LDA()
        clf.fit(X_train_selected, y[train_idx])
        probs = clf.predict_proba(X_test_selected)
        
        # Armazenar resultados
        for i, idx in enumerate(test_idx):
            results.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
            )

    return results
