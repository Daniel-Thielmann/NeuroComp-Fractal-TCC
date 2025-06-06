import os
import pandas as pd
import matplotlib.pyplot as plt

# Garante que a pasta graphics existe
os.makedirs("graphics/results", exist_ok=True)

# Caminho para o CSV final
csv_path = "results/summaries/Fractal_vs_LogPower_comparison.csv"
df = pd.read_csv(csv_path)

# Cria o gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(
    df["LogPower"],
    df["Fractal"],
    alpha=0.5,
    color="purple",
    edgecolor="k",
    label="Trials",
)

# Linha de referência 45°
lims = [0, 1]
plt.plot(lims, lims, "--", color="gray", label="Linha 45° (Igualdade)")

plt.xlabel("Probabilidade correta - LogPower")
plt.ylabel("Probabilidade correta - Fractal")
plt.title("Dispersao: Fractal vs LogPower")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Salva o gráfico
output_path = "graphics/results/scatter_fractal_vs_logpower.png"
plt.savefig(output_path)
plt.close()

print(f"Gráfico de dispersão salvo em: {output_path}")
