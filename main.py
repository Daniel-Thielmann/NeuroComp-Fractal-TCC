"""
Imports
"""
from methods.logpower import LogPower  # já corrigido para evitar erro 'data'
import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.core.kfold import kfold
from bciflow.datasets.cbcic import cbcic
from methods.hig import HiguchiFractal

"""
Script principal para comparação de métodos de extração de características em EEG.
"""


def contar_sujeitos_disponiveis(max_test=20):
    """Identifica quantos sujeitos estão disponíveis no dataset."""
    sujeitos = []
    for i in range(1, max_test + 1):
        try:
            cbcic(subject=i)
            sujeitos.append(i)
        except:
            continue
    return sujeitos


def run_experiment(subject, feature_extractor, feature_name):
    """Executa um experimento completo para um sujeito."""
    try:
        dataset = _load_dataset(subject)
        pre_folding, pos_folding = _create_processing_pipeline(
            feature_extractor)
        start_time = time.time()
        results = kfold(
            target=dataset,
            start_window=dataset['events']['cue'][0] + 0.5,
            pre_folding=pre_folding,
            pos_folding=pos_folding
        )
        elapsed_time = time.time() - start_time
        accuracy = _calculate_accuracy(results)
        print(
            f"Sujeito {subject:02d} | {feature_name:15} | Acurácia: {accuracy * 100:.2f}% | Tempo: {elapsed_time:.1f}s")
        return accuracy
    except Exception as e:
        print(f"Sujeito {subject:02d} | {feature_name:15} | Falhou: {str(e)}")
        return None


def _load_dataset(subject):
    """Carrega o dataset para um sujeito."""
    dataset = cbcic(subject=subject)
    if not dataset or 'events' not in dataset:
        raise ValueError("Dataset inválido ou incompleto")
    return dataset


def _create_processing_pipeline(feature_extractor):
    """Configura os pipelines de processamento."""
    pre_folding = {'tf': (chebyshevII, {})}
    pos_folding = {
        'fe': (feature_extractor, {}),
        'clf': (LDA(), {})
    }
    return pre_folding, pos_folding


def _calculate_accuracy(results):
    """Calcula a acurácia a partir dos resultados."""
    true_labels = np.array(results['true_label'])
    predict_labels = np.array(
        ['left-hand' if i[0] > i[1] else 'right-hand'
         for i in np.array(results)[:, -2:]]
    )
    return accuracy_score(true_labels, predict_labels)


def perform_wilcoxon_test(results_method1, results_method2, method1_name, method2_name):
    """Executa o teste de Wilcoxon para comparação de métodos."""
    print(f"\n=== Teste de Wilcoxon: {method1_name} vs {method2_name} ===")

    # Filtra resultados None (experimentos que falharam)
    res1 = [x for x in results_method1 if x is not None]
    res2 = [x for x in results_method2 if x is not None]

    if not res1 or not res2:
        print("Dados insuficientes para comparação estatística")
        return

    stat, p_value = wilcoxon(res1, res2)

    print(f"Estatística do teste: {stat:.4f}")
    print(f"Valor-p: {p_value:.4f}")

    if p_value > 0.05:
        print("Não há diferença estatisticamente significativa (p > 0.05)")
    else:
        better_method = method1_name if np.mean(
            res1) > np.mean(res2) else method2_name
        print(f"{better_method} é estatisticamente melhor (p < 0.05)")


def display_results(subjects, higuchi_accuracies, logpower_accuracies):
    """Exibe os resultados consolidados."""
    print("\n=== Resultados Consolidados ===")

    print("\nAcurácias por sujeito - Higuchi Fractal:")
    for subj, acc in zip(subjects, higuchi_accuracies):
        status = f"{acc * 100:.2f}%" if acc is not None else "Falhou"
        print(f"Sujeito {subj}: {status}")

    print("\nAcurácias por sujeito - LogPower:")
    for subj, acc in zip(subjects, logpower_accuracies):
        status = f"{acc * 100:.2f}%" if acc is not None else "Falhou"
        print(f"Sujeito {subj}: {status}")

    # Calcula médias apenas para experimentos bem-sucedidos
    higuchi_mean = np.mean([x for x in higuchi_accuracies if x is not None])
    logpower_mean = np.mean([x for x in logpower_accuracies if x is not None])

    print(f"\nMédia Higuchi Fractal: {higuchi_mean * 100:.2f}%")
    print(f"Média LogPower: {logpower_mean * 100:.2f}%")


def main():
    """Fluxo principal de execução."""
    try:
        subjects = contar_sujeitos_disponiveis()

        if not subjects:
            print("Nenhum sujeito encontrado no dataset!")
            return

        print(f"Sujeitos disponíveis: {subjects}")

        extractors = [
            (HiguchiFractal(kmax=10), "Higuchi Fractal"),
            (LogPower(flatting=True), "LogPower")
        ]

        results = {name: [] for _, name in extractors}

        for subject in subjects:
            for extractor, name in extractors:
                accuracy = run_experiment(subject, extractor, name)
                results[name].append(accuracy)

        # Apresentação dos resultados
        display_results(
            subjects, results["Higuchi Fractal"], results["LogPower"]
        )

        # Análise estatística
        perform_wilcoxon_test(
            results["Higuchi Fractal"],
            results["LogPower"],
            "Higuchi Fractal",
            "LogPower"
        )

    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")


if __name__ == "__main__":
    main()
