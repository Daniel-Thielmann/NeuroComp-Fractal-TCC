import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cria a pasta de saída se não existir
os.makedirs("graphics", exist_ok=True)

# Caminho do CSV com os resultados
csv_path = "results/higuchi_vs_logpower_comparison.csv"

# Carrega o CSV
df = pd.read_csv(csv_path)

# Gera o boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, palette="Set2")
plt.title("Distribuição dos Valores - Higuchi vs LogPower")
plt.ylabel("Probabilidade (left/right correta)")
plt.xlabel("Método")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Salva o gráfico
output_path = "graphics/boxplot_higuchi_vs_logpower.png"
plt.savefig(output_path)
plt.close()

print(f"Boxplot salvo em: {output_path}")
