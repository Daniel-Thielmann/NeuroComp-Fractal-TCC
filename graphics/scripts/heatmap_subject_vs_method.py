import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("graphics/results", exist_ok=True)

subjects = [f"P{i:02d}" for i in range(1, 10)]  # Apenas 9 sujeitos
methods = ["Fractal", "LogPower"]
data = {method: [] for method in methods}

for method in methods:
    for i in range(1, 10):  # Apenas 9 sujeitos
        path = f"results/{method}/Training/P{i:02d}.csv"
        try:
            df = pd.read_csv(path)
            df = df[df["true_label"].isin([1, 2])]
            correct_probs = df.apply(
                lambda row: (
                    row["left_prob"] if row["true_label"] == 1 else row["right_prob"]
                ),
                axis=1,
            )
            data[method].append(correct_probs.mean())
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {path}")
            data[method].append(np.nan)  # Adiciona NaN se o arquivo não existir

heatmap_data = pd.DataFrame(data, index=subjects).T

plt.figure(figsize=(10, 2.5))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f",
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
)
plt.title("Heatmap - Media de Probabilidade Correta por Sujeito")
plt.xlabel("Sujeito")
plt.ylabel("Metodo")
plt.tight_layout()

output_path = "graphics/results/heatmap_subject_vs_method.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico salvo em: {output_path}")
