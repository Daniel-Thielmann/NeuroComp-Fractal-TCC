import os
import matplotlib.pyplot as plt

# Garante a pasta de saída
os.makedirs("graphics", exist_ok=True)

# Valor-p obtido pelo teste de Wilcoxon (já conhecido do experimento)
p_value = 0.0063
significance_threshold = 0.05

# Cria o gráfico
plt.figure(figsize=(6, 5))
plt.bar(["Wilcoxon p-value"], [p_value], color="green" if p_value < 0.05 else "gray")
plt.axhline(y=significance_threshold, color='red', linestyle='--', label="Nível de significância (0.05)")
plt.text(0, significance_threshold + 0.005, "p = 0.05", color="red")
plt.title("Resultado do Teste de Wilcoxon")
plt.ylabel("Valor-p")
plt.ylim(0, 0.07)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Salva o gráfico
output_path = "graphics/wilcoxon_pvalue_plot.png"
plt.savefig(output_path)
plt.close()

print(f"Gráfico do valor-p salvo em: {output_path}")
