import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("graphics/results", exist_ok=True)

subjects = [f"P{i:02d}" for i in range(1, 11)]
fractal_means = []
logpower_means = []

for i in range(1, 11):
    fractal_path = f"results/Fractal/Training/P{i:02d}.csv"
    logpower_path = f"results/LogPower/Training/P{i:02d}.csv"

    fractal_df = pd.read_csv(fractal_path)
    log_df = pd.read_csv(logpower_path)

    fractal_df["correct_prob"] = fractal_df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    log_df["correct_prob"] = log_df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )

    fractal_means.append(fractal_df["correct_prob"].mean())
    logpower_means.append(log_df["correct_prob"].mean())

x = np.arange(len(subjects))  # posições dos sujeitos
width = 0.35  # largura das barras

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, fractal_means, width, label="Fractal", color="blue")
plt.bar(x + width/2, logpower_means, width, label="LogPower", color="red")

plt.title("Distribuição da Probabilidade Média Correta por Sujeito")
plt.xlabel("Sujeito")
plt.ylabel("Probabilidade Média Correta")
plt.ylim(0, 1.05)
plt.xticks(x, subjects)
plt.legend()
plt.grid(True, linestyle="--", axis="y", alpha=0.5)
plt.tight_layout()

output_path = "graphics/results/histogram_accuracy_per_subject.png"
plt.savefig(output_path)
plt.close()

print(f"Histograma salvo em: {output_path}")
