import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("graphics/results", exist_ok=True)

df = pd.read_csv("results/summaries/Fractal_vs_LogPower_comparison.csv")

plt.figure(figsize=(10, 6))
plt.hist(
    df["Fractal"], bins=30, alpha=0.6, label="Fractal", color="blue", edgecolor="black"
)
plt.hist(
    df["LogPower"], bins=30, alpha=0.6, label="LogPower", color="red", edgecolor="black"
)

plt.title("Histograma - Distribuição das Probabilidades Corretas")
plt.xlabel("Probabilidade correta (left/right)")
plt.ylabel("Frequência")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

output_path = "graphics/results/histogram_fractal_vs_logpower.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Grafico salvo em: {output_path}")
