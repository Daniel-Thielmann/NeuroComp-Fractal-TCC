import os
import pandas as pd

# Mapeamento dos nomes para as colunas da tabela
method_map = {
    "LogPower": "BNL",
    "CSP_LogPower": "BCL",
    "FBCSP_LogPower": "FCL",
    "Fractal": "BNH",
    "CSP_Fractal": "BCH",
    "FBCSP_Fractal": "FCH",
}

subjects = [f"P{n:02d}" for n in range(1, 10)]
methods = list(method_map.keys())

# Dicionário para armazenar os resultados
results = {subj: {} for subj in subjects}

for method in methods:
    for subj in subjects:
        path = f"results/{method}/Evaluate/{subj}.csv"
        if not os.path.exists(path):
            results[subj][method_map[method]] = ""
            continue
        df = pd.read_csv(path)
        # Predição: 1 se left_prob > right_prob, 2 caso contrário
        pred = (df["left_prob"] < df["right_prob"]).astype(int) + 1
        acc = (pred == df["true_label"]).mean()
        results[subj][method_map[method]] = f"{acc:.3f}"

# Calcular médias
mean_row = {}
for col in method_map.values():
    vals = [float(results[subj][col]) for subj in subjects if results[subj][col] != ""]
    mean_row[col] = f"{sum(vals)/len(vals):.3f}" if vals else ""

# Gerar tabela LaTeX
header = (
    " & ".join(
        ["\\textbf{Sujeito}"] + [f"\\textbf{{{col}}}" for col in method_map.values()]
    )
    + " \\\\ \\hline"
)
lines = []
for subj in subjects:
    line = (
        " & ".join([subj[1:]] + [results[subj][col] for col in method_map.values()])
        + " \\\\"
    )
    lines.append(line)
mean_line = (
    " & ".join(["\\textbf{Média}"] + [mean_row[col] for col in method_map.values()])
    + " \\\\ \\hline"
)

latex = "\\begin{table}[H]\n\\centering\n\\caption{Acurácia por sujeito para cada pipeline.}\n\\label{tab:accuracy_subjects}\n\\begin{tabular}{lcccccc} \\hline\n"
latex += (
    header
    + "\n"
    + "\n".join(lines)
    + "\n"
    + mean_line
    + "\n\\end{tabular}\n\\end{table}"
)

print(latex)
