import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Garante a pasta para salvar o gráfico
os.makedirs("graphics", exist_ok=True)

# Carrega o CSV final com as probabilidades corretas
df = pd.read_csv("results/higuchi_vs_logpower_comparison.csv")

# Reestrutura em formato longo (long format)
df_long = pd.melt(df, value_vars=["Higuchi", "LogPower"],
                  var_name="Método", value_name="Probabilidade_Correta")

# Gera o gráfico de violino com correção do aviso
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_long, x="Método", y="Probabilidade_Correta",
               hue="Método", palette="Set2", inner="quartile", legend=False)

plt.title("Distribuição das Probabilidades Corretas - Higuchi vs LogPower (Violin Plot)")
plt.ylabel("Probabilidade correta (left/right)")
plt.xlabel("Método")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# Salva o gráfico
plt.savefig("graphics/violinplot_higuchi_vs_logpower.png", dpi=300)
plt.close()

print("Violin Plot salvo em: graphics/violinplot_higuchi_vs_logpower.png")
