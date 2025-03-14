import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.core.kfold import kfold
from bciflow.datasets.cbcic import cbcic
from methods.hig import HiguchiFractal

# Carregar o dataset
print("Carregando dataset...")
dataset = cbcic(subject=1)

# Criar instância da classe HiguchiFractal
higuchi_extractor = HiguchiFractal(kmax=10)

# Pipeline de pré e pós-processamento
pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'fe': (higuchi_extractor, {}), 'clf': (LDA(), {})}

# Executar validação cruzada
print("Iniciando validação cruzada K-Fold...")
start_time = time.time()

results = kfold(target=dataset,
                start_window=dataset['events']['cue'][0] + 0.5,
                pre_folding=pre_folding,
                pos_folding=pos_folding)

elapsed_time = time.time() - start_time
print(f"Validação cruzada concluída em {elapsed_time:.2f} segundos.")

# Avaliação de Resultados
true_labels = np.array(results['true_label'])
predict_labels = np.array(
    ['left-hand' if i[0] > i[1] else 'right-hand' for i in np.array(results)[:, -2:]])

accuracy = accuracy_score(true_labels, predict_labels)
conf_matrix = confusion_matrix(true_labels, predict_labels)
class_report = classification_report(true_labels, predict_labels)

# Exibir resultados detalhados
print("\n==== Resultados da Avaliação do Modelo ====")
print(f"Acurácia obtida: {accuracy * 100:.2f}%\n")

# Explicação detalhada da Matriz de Confusão
print("==== Matriz de Confusão ====")
print(conf_matrix, "\n")
print(f"{conf_matrix[0, 0]} previsões corretas para 'Mão Esquerda'")
print(f"{conf_matrix[1, 1]} previsões corretas para 'Mão Direita'")
print(
    f"{conf_matrix[0, 1]} erros onde o modelo deveria prever 'Mão Esquerda', mas disse 'Mão Direita'")
print(
    f"{conf_matrix[1, 0]} erros onde o modelo deveria prever 'Mão Direita', mas disse 'Mão Esquerda'\n")

# Explicação detalhada do Classification Report
print("==== Relatório de Classificação ====")
print(class_report)

print("\nExplicação do Relatório de Classificação:")
print("- Precisão (Precision): Indica quantas previsões para cada classe estavam corretas.")
print("- Revocação (Recall): Mede quantas amostras reais foram corretamente identificadas.")
print("- F1-Score: Média entre Precisão e Recall, mostrando o equilíbrio do modelo.")
print("- Suporte (Support): Quantidade de amostras em cada classe usada para avaliação.")
