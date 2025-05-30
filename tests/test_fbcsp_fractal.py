import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bciflow.datasets import cbcic
from methods.pipelines.fbcsp_fractal import run_fbcsp_fractal


def test_run_fbcsp_fractal_format():
    """Testa o formato de saída da função run_fbcsp_fractal para um sujeito."""
    # Executa o FBCSP+Fractal para um sujeito
    subject_id = 1
    rows = run_fbcsp_fractal(subject_id)
    
    # Verifica se a saída é uma lista de dicionários
    assert isinstance(rows, list), "A saída deve ser uma lista"
    assert all(isinstance(row, dict) for row in rows), "Todos os elementos devem ser dicionários"
    
    # Verifica se cada dicionário tem as chaves corretas
    expected_keys = {"subject_id", "fold", "true_label", "left_prob", "right_prob"}
    for row in rows:
        assert set(row.keys()) == expected_keys, f"Chaves incorretas: {set(row.keys())}"
    
    # Verifica se há 5 folds e se os valores estão corretos
    folds = set(row["fold"] for row in rows)
    assert folds == {0, 1, 2, 3, 4}, f"Folds incorretos: {folds}"
    
    # Verifica se os valores de probabilidade estão entre 0 e 1
    for row in rows:
        assert 0 <= row["left_prob"] <= 1, f"Probabilidade left inválida: {row['left_prob']}"
        assert 0 <= row["right_prob"] <= 1, f"Probabilidade right inválida: {row['right_prob']}"
        assert abs(row["left_prob"] + row["right_prob"] - 1.0) < 1e-6, \
            f"Soma das probabilidades não é 1: {row['left_prob'] + row['right_prob']}"
    
    # Verifica se o subject_id está correto
    assert all(row["subject_id"] == subject_id for row in rows), "Subject ID incorreto"
    
    # Verifica se os rótulos são 1 ou 2
    labels = set(row["true_label"] for row in rows)
    assert labels.issubset({1, 2}), f"Rótulos inválidos: {labels}"


def test_run_fbcsp_fractal_performance():
    """Testa o desempenho do método FBCSP+Fractal em um sujeito."""
    # Executa o FBCSP+Fractal para um sujeito
    subject_id = 1
    rows = run_fbcsp_fractal(subject_id)
    
    # Converte para DataFrame para facilitar análises
    df = pd.DataFrame(rows)
    
    # Calcula acurácia
    df["predicted"] = (df["left_prob"] < 0.5).astype(int) + 1
    accuracy = (df["true_label"] == df["predicted"]).mean()
    
    # Verifica se a acurácia está acima do nível de chance (50%)
    assert accuracy > 0.5, f"Acurácia abaixo do nível de chance: {accuracy}"
    print(f"Acurácia do FBCSP+Fractal para o sujeito {subject_id}: {accuracy:.4f}")


def run_fbcsp_fractal_all_subjects():
    """Executa o método FBCSP+Fractal para todos os sujeitos e retorna os resultados."""
    all_rows = []
    
    for subject_id in tqdm(range(1, 10), desc="FBCSP_Fractal"):
        try:
            rows = run_fbcsp_fractal(subject_id)
            all_rows.extend(rows)
            
            # Salva os resultados para este sujeito
            df_subject = pd.DataFrame(rows)
            
            # Divide em conjuntos de treinamento e avaliação
            os.makedirs("results/FBCSP_Fractal/Training", exist_ok=True)
            os.makedirs("results/FBCSP_Fractal/Evaluate", exist_ok=True)
            
            df_subject[df_subject["fold"] < 4].to_csv(
                f"results/FBCSP_Fractal/Training/P{subject_id:02d}.csv", index=False
            )
            df_subject[df_subject["fold"] == 4].to_csv(
                f"results/FBCSP_Fractal/Evaluate/P{subject_id:02d}.csv", index=False
            )
            
        except Exception as e:
            print(f"Erro ao processar o sujeito {subject_id}: {str(e)}")
    
    return pd.DataFrame(all_rows)


def test_fbcsp_fractal_all_subjects_performance():
    """Testa o desempenho do método FBCSP+Fractal em todos os sujeitos."""
    df = run_fbcsp_fractal_all_subjects()
    
    # Calcula métricas de desempenho
    df["predicted"] = (df["left_prob"] < 0.5).astype(int) + 1
    accuracy = (df["true_label"] == df["predicted"]).mean()
    
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1
    )
    mean_correct_prob = df["correct_prob"].mean()
    
    # Calcula acurácia por sujeito
    subject_accuracies = df.groupby("subject_id").apply(
        lambda x: (x["true_label"] == x["predicted"]).mean()
    )
    
    print(f"Acurácia global: {accuracy:.4f}")
    print(f"Média de probabilidade correta: {mean_correct_prob:.4f}")
    print("Acurácia por sujeito:")
    for subject_id, acc in subject_accuracies.items():
        print(f"  Sujeito {subject_id}: {acc:.4f}")


if __name__ == "__main__":
    # Executa os testes unitários
    test_run_fbcsp_fractal_format()
    test_run_fbcsp_fractal_performance()
    print("Todos os testes unitários passaram!")
    
    # Executa o teste de desempenho para todos os sujeitos
    test_fbcsp_fractal_all_subjects_performance()
