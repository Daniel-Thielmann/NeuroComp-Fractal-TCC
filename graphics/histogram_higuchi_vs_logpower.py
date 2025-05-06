import os
import pandas as pd
import matplotlib.pyplot as plt

# Garante que a pasta de gráficos exista
os.makedirs("graphics", exist_ok=True)

# Caminho do CSV comparativo
csv_path = "results/higuchi_vs_logpower_comparison.csv"

# Carrega os dados
df = pd.read_csv(csv_path)

# Cria o histograma sobreposto
plt.figure(figsize=(8, 6))
plt.hist(df["Higuchi"], bins=30, alpha=0.6, label="Higuchi", color="skyblue", edgecolor="black")
plt.hist(df["LogPower"], bins=30, alpha=0.6, label="LogPower", color="salmon", edgecolor="black")

plt.title("Histograma das Probabilidades - Higuchi vs LogPower")
plt.xlabel("Probabilidade correta (left/right)")
plt.ylabel("Frequência")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Salva o gráfico
output_path = "graphics/histogram_higuchi_vs_logpower.png"
plt.savefig(output_path)
plt.close()

print(f"Histograma salvo em: {output_path}")
