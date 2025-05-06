import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("graphics/results", exist_ok=True)

df = pd.read_csv("results/higuchi_vs_logpower_comparison.csv")
df_long = pd.melt(df, value_vars=["Higuchi", "LogPower"],
                  var_name="Método", value_name="Probabilidade_Correta")

plt.figure(figsize=(8, 6))
sns.violinplot(data=df_long, x="Método", y="Probabilidade_Correta",
               hue="Método", palette="Set2", inner="quartile", legend=False)

plt.title("Violin Plot - Higuchi vs LogPower")
plt.ylabel("Probabilidade correta (left/right)")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

output_path = "graphics/results/violinplot_higuchi_vs_logpower.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico salvo em: {output_path}")
