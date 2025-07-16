import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

os.makedirs("results/comprehensive_results", exist_ok=True)

# Configuração dos datasets disponíveis
DATASETS_CONFIG = {
    "wcci2020": {
        "name": "WCCI2020",
        "subjects": 9,
        "description": "WCCI 2020 Competition Dataset (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.test_fractal.test_fractal_classification_wcci2020",
            "logpower": "tests.test_logpower.test_logpower_classification_wcci2020",
            "csp_fractal": "tests.test_csp_fractal.test_csp_fractal_classification_wcci2020",
            "csp_logpower": "tests.test_csp_logpower.test_csp_logpower_pipeline",
            "fbcsp_pure": "tests.test_fbcsp_pure.test_fbcsp_pure_pipeline",
            "fbcsp_logpower": "tests.test_fbcsp_logpower.test_fbcsp_logpower_pipeline",
            "fbcsp_fractal": "tests.test_fbcsp_fractal.test_fbcsp_fractal_classification_wcci2020",
        },
    },
    "bciciv2a": {
        "name": "BCI Competition IV Dataset 2a",
        "subjects": 9,
        "description": "BCI Competition IV Dataset 2a (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.test_fractal.test_fractal_classification_bciciv2a",
            "logpower": "tests.test_logpower.test_logpower_classification_bciciv2a",
            "csp_fractal": "tests.test_csp_fractal.test_csp_fractal_classification_bciciv2a",
            "csp_logpower": "tests.test_csp_logpower.test_csp_logpower_classification_bciciv2a",
            "fbcsp_pure": "tests.test_fbcsp_pure.test_fbcsp_pure_classification_bciciv2a",
            "fbcsp_logpower": "tests.test_fbcsp_logpower.test_fbcsp_logpower_classification_bciciv2a",
            "fbcsp_fractal": "tests.test_fbcsp_fractal.test_fbcsp_fractal_classification_bciciv2a",
        },
    },
    "bciciv2b": {
        "name": "BCI Competition IV Dataset 2b",
        "subjects": 9,
        "description": "BCI Competition IV Dataset 2b (9 sujeitos)",
        "test_functions": {
            "fractal": "tests.test_fractal.test_fractal_classification_bciciv2b",
            "logpower": "tests.test_logpower.test_logpower_classification_bciciv2b",
            "csp_fractal": "tests.test_csp_fractal.test_csp_fractal_classification_bciciv2b",
            "csp_logpower": "tests.test_csp_logpower.test_csp_logpower_classification_bciciv2b",
            "fbcsp_pure": "tests.test_fbcsp_pure.test_fbcsp_pure_classification_bciciv2b",
            "fbcsp_logpower": "tests.test_fbcsp_logpower.test_fbcsp_logpower_classification_bciciv2b",
            "fbcsp_fractal": "tests.test_fbcsp_fractal.test_fbcsp_fractal_classification_bciciv2b",
        },
    },
}


def list_available_datasets():
    """Lista todos os datasets disponíveis."""
    print("📋 DATASETS DISPONÍVEIS:")
    print("-" * 50)
    for key, config in DATASETS_CONFIG.items():
        print(f"🔹 {key}: {config['name']}")
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
    print(f"🧠 TESTE COMPLETO DE CLASSIFICAÇÃO EEG - {config['name'].upper()}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {config['name']} ({config['subjects']} sujeitos)")
    print(f"Descrição: {config['description']}")
    print("Tarefa: Classificação de motor imagery (left-hand vs right-hand)")
    print("Validação: 5-fold Cross Validation")
    print("-" * 80)

    # Define métodos a executar
    available_methods = {
        "fractal": {
            "name": "Fractal Puro",
            "pipeline": "Filtro 8-30Hz → Higuchi Fractal → StandardScaler → LDA",
            "emoji": "🔬",
        },
        "logpower": {
            "name": "LogPower Puro",
            "pipeline": "Filtro 8-30Hz → Log(Mean(X²)) → StandardScaler → LDA",
            "emoji": "⚡",
        },
        "csp_fractal": {
            "name": "CSP + Fractal",
            "pipeline": "Filtro 8-30Hz → CSP (4 comp) → Higuchi Fractal → StandardScaler → LDA",
            "emoji": "🔄",
        },
        "csp_logpower": {
            "name": "CSP + LogPower",
            "pipeline": "Filtro 8-30Hz → CSP (4 comp) → Log(Var) → StandardScaler → LDA",
            "emoji": "🔄",
        },
        "fbcsp_pure": {
            "name": "FBCSP + Pure",
            "pipeline": "Filter Bank → CSP (2 comp extremos) → Log(Var) → MIBIF → StandardScaler → LDA",
            "emoji": "🏆",
        },
        "fbcsp_logpower": {
            "name": "FBCSP + LogPower",
            "pipeline": "Filter Bank → CSP (4 comp) → Log(Energy) → MIBIF → StandardScaler → LDA",
            "emoji": "🏆",
        },
        "fbcsp_fractal": {
            "name": "FBCSP + Fractal",
            "pipeline": "Filter Bank → CSP (4 comp) → [Fractal+Energy+Std] → MIBIF → StandardScaler → LDA",
            "emoji": "🏆",
        },
    }

    # Se métodos específicos foram solicitados, filtra apenas eles
    if methods is not None:
        methods_to_run = {k: v for k, v in available_methods.items() if k in methods}
        if not methods_to_run:
            raise ValueError(
                f"Nenhum método válido especificado. Métodos disponíveis: {list(available_methods.keys())}"
            )
    else:
        methods_to_run = available_methods

    all_results = {}

    # Executa cada método
    for i, (method_key, method_info) in enumerate(methods_to_run.items(), 1):
        print(f"\n{method_info['emoji']} {i}. TESTANDO: {method_info['name']}")
        print(f"Pipeline: {method_info['pipeline']}")

        try:
            # Importa e executa a função de teste do método
            if method_key in config["test_functions"]:
                test_function = import_function_from_string(
                    config["test_functions"][method_key]
                )
                results = test_function()
                all_results[f"{method_info['name'].replace(' ', '_')}"] = (
                    extract_summary_stats(results)
                )
                print(f"✅ {method_info['name']} concluído")
            else:
                print(
                    f"⚠️  {method_info['name']} não implementado para dataset {dataset}"
                )
                all_results[f"{method_info['name'].replace(' ', '_')}"] = {
                    "error": f"Não implementado para {dataset}"
                }

        except Exception as e:
            print(f"❌ Erro no {method_info['name']}: {e}")
            all_results[f"{method_info['name'].replace(' ', '_')}"] = {"error": str(e)}

    # RESULTADOS FINAIS
    print("\n" + "=" * 80)
    print(f"📊 RESULTADOS FINAIS - RANKING DE PERFORMANCE ({config['name']})")
    print("=" * 80)

    # Prepara dados para ranking
    valid_results = []
    for method, stats in all_results.items():
        if "error" not in stats and "mean_accuracy" in stats:
            valid_results.append(
                {
                    "Método": method,
                    "Acurácia Média (%)": f"{stats['mean_accuracy']*100:.2f}",
                    "Desvio Padrão (%)": f"{stats['std_accuracy']*100:.2f}",
                    "Kappa Médio": f"{stats['mean_kappa']:.4f}",
                    "Melhor (%)": f"{stats['max_accuracy']*100:.2f}",
                    "Pior (%)": f"{stats['min_accuracy']*100:.2f}",
                }
            )

    # Ordena por acurácia média (decrescente)
    valid_results.sort(
        key=lambda x: float(x["Acurácia Média (%)"].replace("%", "")), reverse=True
    )

    # Exibe ranking
    print(
        f"{'Rank':<4} {'Método':<18} {'Acurácia':<12} {'±Desvio':<10} {'Kappa':<8} {'Melhor':<8} {'Pior':<8}"
    )
    print("-" * 75)

    for i, result in enumerate(valid_results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(
            f"{emoji}{i:<3} {result['Método']:<18} {result['Acurácia Média (%)']:<12} "
            f"±{result['Desvio Padrão (%)']:<9} {result['Kappa Médio']:<8} "
            f"{result['Melhor (%)']:<8} {result['Pior (%)']:<8}"
        )

    # Exibe erros se houver
    errors = [method for method, stats in all_results.items() if "error" in stats]
    if errors:
        print(f"\n❌ MÉTODOS COM ERRO: {', '.join(errors)}")

    # Salva resultados em CSV
    save_comprehensive_results(all_results, valid_results, dataset)

    print("\n✅ TESTE COMPLETO FINALIZADO!")
    print(f"📁 Resultados salvos em: results/comprehensive_results/{dataset}/")

    return all_results


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
            "fbcsp_pure",
            "fbcsp_logpower",
            "fbcsp_fractal",
        ],
        help="Métodos específicos a executar (padrão: todos):\n"
        "  fractal: Fractal puro\n"
        "  logpower: LogPower puro\n"
        "  csp_fractal: CSP + Fractal\n"
        "  csp_logpower: CSP + LogPower\n"
        "  fbcsp_pure: FBCSP + Pure\n"
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


def main():
    """Função principal que executa todos os testes."""
    try:
        args = parse_arguments()

        # Lista datasets se solicitado
        if args.list_datasets:
            list_available_datasets()
            return

        # Valida dataset
        if args.dataset not in DATASETS_CONFIG:
            print(f"❌ Erro: Dataset '{args.dataset}' não encontrado.")
            list_available_datasets()
            return

        # Executa testes
        if not args.quiet:
            print(f"🚀 Iniciando testes com dataset: {args.dataset}")
            if args.methods:
                print(f"📝 Métodos específicos: {', '.join(args.methods)}")
            print()

        all_results = run_all_eeg_tests(dataset=args.dataset, methods=args.methods)

        # Exibe resumo final simples
        if not args.quiet:
            print("\n" + "=" * 50)
            print("📋 RESUMO EXECUTIVO")
            print("=" * 50)

            valid_methods = 0
            error_methods = 0

            for method, stats in all_results.items():
                if "error" in stats:
                    error_methods += 1
                else:
                    valid_methods += 1

            print(f"✅ Métodos executados com sucesso: {valid_methods}")
            print(f"❌ Métodos com erro: {error_methods}")
            print(f"📊 Total de métodos testados: {len(all_results)}")
            print(f"🗂️  Dataset usado: {DATASETS_CONFIG[args.dataset]['name']}")

            if valid_methods > 0:
                # Encontra melhor método
                best_method = None
                best_accuracy = 0

                for method, stats in all_results.items():
                    if "error" not in stats and "mean_accuracy" in stats:
                        if stats["mean_accuracy"] > best_accuracy:
                            best_accuracy = stats["mean_accuracy"]
                            best_method = method

                if best_method:
                    print(f"🏆 Melhor método: {best_method} ({best_accuracy*100:.2f}%)")

        return all_results

    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ Erro crítico na execução: {e}")
        import traceback

        traceback.print_exc()


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


if __name__ == "__main__":
    main()
