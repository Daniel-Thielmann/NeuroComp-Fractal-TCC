import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATASETS = [
    ("wcci2020", "WCCI2020"),
    ("bciciv2a", "BCICIV2a"),
    ("bciciv2b", "BCICIV2b"),
]

METHODS = {
    "fractal": "fractal",
    "logpower": "logpower",
    "csp_fractal": "csp_fractal",
    "csp_logpower": "csp_logpower",
    "fbcsp_fractal": "fbcsp_fractal",
    "fbcsp_logpower": "fbcsp_logpower",
}

COMPARISONS = [
    ("fractal", "logpower", "Fractal vs. LogPower"),
    ("csp_fractal", "csp_logpower", "CSP+Fractal vs. CSP+LogPower"),
    ("fbcsp_fractal", "fbcsp_logpower", "FBCSP+Fractal vs. FBCSP+LogPower"),
]


def get_metric(dataset, method, metric_name):
    values = []
    method_dir = os.path.join(f"d:/dev/EEG-TCC/results/{dataset}", method, "evaluate")
    for i in range(1, 10):
        csv_path = os.path.join(method_dir, f"P{i:02d}_evaluate.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if metric_name.lower() == "accuracy":
            if "Accuracy" in df.columns:
                val = df["Accuracy"].mean()
            elif "Test_Accuracy" in df.columns:
                val = df["Test_Accuracy"].mean()
            elif "accuracy" in df.columns:
                val = df["accuracy"].mean()
            else:
                continue
        elif metric_name.lower() == "kappa":
            if "Kappa" in df.columns:
                val = df["Kappa"].mean()
            elif "Test_Kappa" in df.columns:
                val = df["Test_Kappa"].mean()
            elif "kappa" in df.columns:
                val = df["kappa"].mean()
            else:
                continue
        else:
            continue
        values.append(val)
    return np.array(values)


def print_all_p_values():
    print(
        "\nTabela de comparação estatística (p-valores) entre pipelines para cada dataset:"
    )
    print("| Comparison                      | WCCI2020   | BCICIV2a   | BCICIV2b   |")
    print("|----------------------------------|------------|------------|------------|")
    for m1, m2, label in COMPARISONS:
        row = f"| {label:<32} |"
        for ds_key, ds_name in DATASETS:
            acc1 = get_metric(ds_key, METHODS[m1], "Accuracy")
            acc2 = get_metric(ds_key, METHODS[m2], "Accuracy")
            if len(acc1) == 0 or len(acc2) == 0:
                pval = "N/A"
            else:
                try:
                    pval = f"{wilcoxon(acc1, acc2).pvalue:.4f}"
                except Exception:
                    pval = "Erro"
            row += f" {pval:<10} |"
        print(row)


if __name__ == "__main__":
    print_all_p_values()
