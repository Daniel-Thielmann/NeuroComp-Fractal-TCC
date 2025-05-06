import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Garante que a pasta de gráficos existe
os.makedirs("graphics", exist_ok=True)

# Carrega o CSV final com as probabilidades corretas
df = pd.read_csv("results/higuchi_vs_logpower_comparison.csv")

# Associa cada 40 trials a um sujeito (40 sujeitos → 400 linhas por método)
df["subject"] = ["P{:02d}".format(i // 40 + 1) for i in range(len(df))]

# Agrupa por sujeito e calcula a média por método
grouped = df.groupby("subject").agg({
    "Higuchi": "mean",
    "LogPower": "mean"
}).reset_index()

# Transforma para formato long para o heatmap
heatmap_data = grouped.set_index("subject").T

# Gera o heatmap com melhorias visuais
plt.figure(figsize=(20, 5))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    cbar=True,
    fmt=".2f",
    linewidths=0.3,
    linecolor='gray'
)

plt.title("Heatmap: Média por Sujeito - Higuchi vs LogPower", fontsize=14)
plt.ylabel("Método", fontsize=12)
plt.xlabel("Sujeito", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Salva o gráfico
plt.savefig("graphics/heatmap_subject_vs_method.png", dpi=300)
plt.close()

print("Gráfico salvo em: graphics/heatmap_subject_vs_method.png")
