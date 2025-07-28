import os

# Garante que a pasta de resultados exista
os.makedirs("graphics/results", exist_ok=True)

# Lista de scripts de geração de gráficos (sem emojis, nomes exatos)
scripts = [
    "accuracy_per_subject.py",
    "boxplot_fractal_vs_logpower.py",  # Atualizado
    "confusion_matrix_comparison.py",
    "heatmap_subject_vs_method.py",
    "histogram_fractal_vs_logpower.py",  # Atualizado
    "scatter_fractal_vs_logpower.py",  # Atualizado
    "violinplot_fractal_vs_logpower.py",  # Atualizado
    "wilcoxon_pvalue_plot.py",
]

# Executa cada script individualmente
for script in scripts:
    script_path = os.path.join("graphics", "scripts", script)
    if os.path.exists(script_path):
        print(f"Executando: {script}")
        os.system(f"python {script_path}")
    else:
        print(f"Script não encontrado: {script}")
