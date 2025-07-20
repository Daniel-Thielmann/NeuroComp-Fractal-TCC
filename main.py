import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


DATASETS_CONFIG = {
    "wcci2020": {
        "name": "WCCI2020",
        "subjects": 9,
        "description": "WCCI 2020 Competition Dataset (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.wcci2020.test_fractal.test_fractal_classification_wcci2020",
            "logpower": "tests.wcci2020.test_logpower.test_logpower_classification_wcci2020",
            "csp_fractal": "tests.wcci2020.test_csp_fractal.test_csp_fractal_classification_wcci2020",
            "csp_logpower": "tests.wcci2020.test_csp_logpower.test_csp_logpower_pipeline",
            "fbcsp_logpower": "tests.wcci2020.test_fbcsp_logpower.test_fbcsp_logpower_pipeline",
            "fbcsp_fractal": "tests.wcci2020.test_fbcsp_fractal.test_fbcsp_fractal_classification_wcci2020",
        },
    },
    "bciciv2a": {
        "name": "BCI Competition IV Dataset 2a",
        "subjects": 9,
        "description": "BCI Competition IV Dataset 2a (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.bciciv2a.test_fractal.test_fractal_classification_bciciv2a",
            "logpower": "tests.bciciv2a.test_logpower.test_logpower_classification_bciciv2a",
            "csp_fractal": "tests.bciciv2a.test_csp_fractal.test_csp_fractal_classification_bciciv2a",
            "csp_logpower": "tests.bciciv2a.test_csp_logpower.test_csp_logpower_classification_bciciv2a",
            "fbcsp_logpower": "tests.bciciv2a.test_fbcsp_logpower.test_fbcsp_logpower_classification_bciciv2a",
            "fbcsp_fractal": "tests.bciciv2a.test_fbcsp_fractal.test_fbcsp_fractal_classification_bciciv2a",
        },
    },
    "bciciv2b": {
        "name": "BCI Competition IV Dataset 2b",
        "subjects": 9,
        "description": "BCI Competition IV Dataset 2b (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.bciciv2b.test_fractal.test_fractal_classification_bciciv2b",
            "logpower": "tests.bciciv2b.test_logpower.test_logpower_classification_bciciv2b",
            "csp_fractal": "tests.bciciv2b.test_csp_fractal.test_csp_fractal_classification_bciciv2b",
            "csp_logpower": "tests.bciciv2b.test_csp_logpower.test_csp_logpower_classification_bciciv2b",
            "fbcsp_logpower": "tests.bciciv2b.test_fbcsp_logpower.test_fbcsp_logpower_classification_bciciv2b",
            "fbcsp_fractal": "tests.bciciv2b.test_fbcsp_fractal.test_fbcsp_fractal_classification_bciciv2b",
        },
    },
}


def list_available_datasets():
    """Lista todos os datasets disponíveis."""
    print("DATASETS DISPONÍVEIS:")
    print("-" * 50)
    for key, config in DATASETS_CONFIG.items():
        print(f"- {key}: {config['name']}")
        print(f"   {config['description']}")
        print()


def import_function_from_string(function_path):
    """
    Importa uma função a partir de uma string no formato 'module.function'.

    Args:
        function_path (str): Caminho da função no formato 'module.submodule.function'

    Returns:
        function: Função importada
    """
    module_path, function_name = function_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


def run_all_eeg_tests(dataset="wcci2020", methods=None):
    """
    Executa todos os testes de classificação EEG padronizados no dataset especificado.

    Args:
        dataset (str): Nome do dataset a ser usado ('wcci2020', 'bciciv2a', 'bciciv2b')
        methods (list): Lista específica de métodos a executar (None = todos)

    Métodos disponíveis:
    1. fractal - Fractal puro
    2. logpower - LogPower puro
    3. csp_fractal - CSP + Fractal
    4. csp_logpower - CSP + LogPower
    5. fbcsp_pure - FBCSP + Pure
    6. fbcsp_logpower - FBCSP + LogPower
    7. fbcsp_fractal - FBCSP + Fractal
    """
    if dataset not in DATASETS_CONFIG:
        raise ValueError(
            f"Dataset '{dataset}' não encontrado. Datasets disponíveis: {list(DATASETS_CONFIG.keys())}"
        )

    config = DATASETS_CONFIG[dataset]

    print("=" * 80)
    print(f"TESTE COMPLETO DE CLASSIFICAÇÃO EEG - {config['name'].upper()}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {config['name']} ({config['subjects']} sujeitos)")
    print(f"Descrição: {config['description']}")
    print("Tarefa: Classificação de motor imagery (left-hand vs right-hand)")
    print("Validação: 5-fold Cross Validation")
    print("-" * 80)

    # Executa os métodos definidos para o dataset e coleta os resultados
    results = {}
    selected_methods = (
        methods if methods is not None else list(config["test_functions"].keys())
    )
    for method in selected_methods:
        if method not in config["test_functions"]:
            results[method] = {
                "error": f"Método '{method}' não disponível para este dataset."
            }
            continue
        func_path = config["test_functions"][method]
        try:
            func = import_function_from_string(func_path)
            method_results = func()
            # Garante que os resultados estejam pareados e completos
            if not isinstance(method_results, dict):
                results[method] = {"error": "Resultado do método não é um dicionário."}
                continue
            # Verifica se todos os sujeitos estão presentes
            n_subjects = config["subjects"]
            missing_subjects = [
                f"P{idx:02d}"
                for idx in range(1, n_subjects + 1)
                if f"P{idx:02d}" not in method_results
            ]
            if missing_subjects:
                results[method] = {"error": f"Sujeitos ausentes: {missing_subjects}"}
                continue
            results[method] = method_results
        except Exception as e:
            results[method] = {"error": str(e)}
    return results


def extract_summary_stats(results):
    """
    Extrai estatísticas resumidas dos resultados de um método.

    Args:
        results (dict): Resultados do método por sujeito

    Returns:
        dict: Estatísticas resumidas (média, desvio, min, max)
    """
    accuracies = []
    kappas = []

    for subject, metrics in results.items():
        if "error" not in metrics and "accuracy" in metrics:
            accuracies.append(metrics["accuracy"])
            kappas.append(metrics["kappa"])

    if not accuracies:
        return {"error": "Nenhum resultado válido"}

    return {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "min_accuracy": np.min(accuracies),
        "max_accuracy": np.max(accuracies),
        "mean_kappa": np.mean(kappas),
        "std_kappa": np.std(kappas),
        "n_subjects": len(accuracies),
    }


def save_comprehensive_results(all_results, ranking_results, dataset="wcci2020"):
    """
    Salva todos os resultados em arquivos CSV organizados.

    Args:
        all_results (dict): Resultados completos por método
        ranking_results (list): Resultados ordenados por performance
        dataset (str): Nome do dataset usado
    """
    # Cria diretório de resultados específico para o dataset
    results_dir = f"results/comprehensive_results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Salva ranking resumido
    ranking_df = pd.DataFrame(ranking_results)
    ranking_df.to_csv(f"{results_dir}/ranking_summary.csv", index=False)

    # 2. Salva estatísticas detalhadas
    detailed_stats = []
    for method, stats in all_results.items():
        if "error" not in stats and "mean_accuracy" in stats:
            detailed_stats.append(
                {
                    "Dataset": dataset,
                    "Método": method,
                    "Acurácia_Média": stats["mean_accuracy"],
                    "Acurácia_Desvio": stats["std_accuracy"],
                    "Acurácia_Min": stats["min_accuracy"],
                    "Acurácia_Max": stats["max_accuracy"],
                    "Kappa_Médio": stats["mean_kappa"],
                    "Kappa_Desvio": stats["std_kappa"],
                    "N_Sujeitos": stats["n_subjects"],
                }
            )

    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv(f"{results_dir}/detailed_statistics.csv", index=False)

    # 3. Salva resultados brutos para análise posterior
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import json

    with open(f"{results_dir}/raw_results_{timestamp}.json", "w") as f:
        # Converte numpy arrays para listas para serialização JSON
        serializable_results = {
            "dataset": dataset,
            "timestamp": timestamp,
            "results": {},
        }
        for method, data in all_results.items():
            if isinstance(data, dict):
                serializable_results["results"][method] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in data.items()
                }
            else:
                serializable_results["results"][method] = data
        json.dump(serializable_results, f, indent=2)


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema de teste de classificação EEG com múltiplos datasets e métodos",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="wcci2020",
        choices=list(DATASETS_CONFIG.keys()),
        help="Dataset a ser usado para os testes:\n"
        + "\n".join([f"  {k}: {v['description']}" for k, v in DATASETS_CONFIG.items()]),
    )

    parser.add_argument(
        "--methods",
        "-m",
        type=str,
        nargs="+",
        choices=[
            "fractal",
            "logpower",
            "csp_fractal",
            "csp_logpower",
            "fbcsp_logpower",
            "fbcsp_fractal",
        ],
        help="Métodos específicos a executar (padrão: todos):\n"
        "  fractal: Fractal puro\n"
        "  logpower: LogPower puro\n"
        "  csp_fractal: CSP + Fractal\n"
        "  csp_logpower: CSP + LogPower\n"
        "  fbcsp_logpower: FBCSP + LogPower\n"
        "  fbcsp_fractal: FBCSP + Fractal",
    )

    parser.add_argument(
        "--list-datasets",
        "-l",
        action="store_true",
        help="Lista todos os datasets disponíveis e sai",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Modo silencioso (reduz output)"
    )

    return parser.parse_args()


# Função auxiliar para compatibilidade com uso direto
def run_wcci2020_tests():
    """Executa testes especificamente no dataset WCCI2020 (compatibilidade)."""
    return run_all_eeg_tests(dataset="wcci2020")


def run_bciciv2a_tests():
    """Executa testes especificamente no dataset BCI Competition IV 2a."""
    return run_all_eeg_tests(dataset="bciciv2a")


def run_bciciv2b_tests():
    """Executa testes especificamente no dataset BCI Competition IV 2b."""
    return run_all_eeg_tests(dataset="bciciv2b")


# Função principal interativa
def main():
    try:
        print("Selecione o dataset para rodar os testes:")
        print("[w] WCCI2020")
        print("[a] BCICIV2a")
        print("[b] BCICIV2b")
        dataset_input = (
            input("Digite a letra correspondente ao dataset (w/a/b): ").strip().lower()
        )
        if dataset_input == "w":
            dataset = "wcci2020"
        elif dataset_input == "a":
            dataset = "bciciv2a"
        elif dataset_input == "b":
            dataset = "bciciv2b"
        else:
            print("[ERRO] Opção inválida. Use 'w', 'a' ou 'b'.")
            return

        print(f"Iniciando testes com dataset: {DATASETS_CONFIG[dataset]['name']}")
        all_results_raw = run_all_eeg_tests(dataset=dataset)

        # Extrai estatísticas resumidas para cada método
        all_results = {}
        for method, stats in all_results_raw.items():
            if "error" not in stats:
                summary = extract_summary_stats(stats)
                stats.update(summary)
            all_results[method] = stats

        print("\n" + "=" * 50)
        print("RESUMO EXECUTIVO")
        print("=" * 50)

        valid_methods = 0
        error_methods = 0
        failed_methods = []
        for method, stats in all_results.items():
            if "error" in stats:
                error_methods += 1
                failed_methods.append((method, stats["error"]))
            else:
                valid_methods += 1

        print(f"[OK] Métodos executados com sucesso: {valid_methods}")
        print(f"[ERRO] Métodos com erro: {error_methods}")
        print(f"Total de métodos testados: {len(all_results)}")
        print(f"Dataset usado: {DATASETS_CONFIG[dataset]['name']}")

        if failed_methods:
            print("\nMétodos que falharam:")
            for method, msg in failed_methods:
                print(f"- {method}: {msg}")

        # Salva resultados em CSVs conforme padrão: results/dataset/metodo/evaluate e results/dataset/metodo/training
        for method, stats in all_results.items():
            if "error" not in stats:
                eval_dir = f"results/{dataset}/{method}/evaluate"
                train_dir = f"results/{dataset}/{method}/training"
                os.makedirs(eval_dir, exist_ok=True)
                os.makedirs(train_dir, exist_ok=True)
                eval_rows = []
                train_rows = []
                for subject, metrics in stats.items():
                    if isinstance(metrics, dict) and "error" not in metrics:
                        # Garante nome P01, P02, ... para todos os prints e CSVs
                        subj_num = "".join(filter(str.isdigit, subject))
                        subject_id = f"P{int(subj_num):02d}" if subj_num else subject
                        # Salva CSV de avaliação (test)
                        eval_df = pd.DataFrame(
                            {
                                "Subject": [subject_id],
                                "Accuracy": [metrics.get("accuracy", None)],
                                "Kappa": [metrics.get("kappa", None)],
                                "N_Samples": [metrics.get("n_samples", None)],
                            }
                        )
                        eval_df.to_csv(
                            f"{eval_dir}/{subject_id}_evaluate.csv", index=False
                        )
                        eval_rows.append(eval_df.iloc[0])
                        # Salva CSV de treinamento (train)
                        train_df = pd.DataFrame(
                            {
                                "Subject": [subject_id],
                                "Accuracy": [metrics.get("train_accuracy", None)],
                                "Kappa": [metrics.get("train_kappa", None)],
                                "N_Samples": [metrics.get("n_samples", None)],
                            }
                        )
                        train_df.to_csv(
                            f"{train_dir}/{subject_id}_training.csv", index=False
                        )
                        train_rows.append(train_df.iloc[0])
                # Salva CSV geral do método (soma dos sujeitos) na pasta do método
                if eval_rows:
                    eval_all_df = pd.DataFrame(eval_rows)
                    eval_all_df.to_csv(
                        f"results/{dataset}/{method}/evaluate_results.csv", index=False
                    )
                if train_rows:
                    train_all_df = pd.DataFrame(train_rows)
                    train_all_df.to_csv(
                        f"results/{dataset}/{method}/training_results.csv", index=False
                    )

        # Mostra o melhor método
        if valid_methods > 0:
            best_method = None
            best_accuracy = 0
            for method, stats in all_results.items():
                if "error" not in stats and "mean_accuracy" in stats:
                    if stats["mean_accuracy"] > best_accuracy:
                        best_accuracy = stats["mean_accuracy"]
                        best_method = method
            if best_method:
                print(f"Melhor método: {best_method} ({best_accuracy*100:.2f}%)")

        return all_results

    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\n[ERRO] Erro crítico na execução: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
