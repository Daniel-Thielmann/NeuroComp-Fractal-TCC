import matplotlib.pyplot as plt
import pandas as pd
import os


# Buscar resultados na pasta do projeto
base_path = r"d:\dev\EEG-TCC\results\wcci2020"

fractal_eval_path = os.path.join(base_path, "Fractal", "Evaluate")
fractal_train_path = os.path.join(base_path, "Fractal", "Training")
logpower_eval_path = os.path.join(base_path, "LogPower", "Evaluate")
logpower_train_path = os.path.join(base_path, "LogPower", "Training")


subjects_eval = [f"P0{i}_evaluate.csv" for i in range(1, 10)]
subjects_train = [f"P0{i}_training.csv" for i in range(1, 10)]


fractal_acc = []
logpower_acc = []


# Adiciona acurácias dos CSVs de Evaluate
for subj in subjects_eval:
    f_file = os.path.join(fractal_eval_path, subj)
    l_file = os.path.join(logpower_eval_path, subj)
    if os.path.exists(f_file) and os.path.exists(l_file):
        f_df = pd.read_csv(f_file)
        l_df = pd.read_csv(l_file)
        fractal_acc.append(float(f_df["Accuracy"].values[0]))
        logpower_acc.append(float(l_df["Accuracy"].values[0]))

# Adiciona acurácias dos CSVs de Training
for subj in subjects_train:
    f_file = os.path.join(fractal_train_path, subj)
    l_file = os.path.join(logpower_train_path, subj)
    if os.path.exists(f_file) and os.path.exists(l_file):
        f_df = pd.read_csv(f_file)
        l_df = pd.read_csv(l_file)
        fractal_acc.append(float(f_df["Accuracy"].values[0]))
        logpower_acc.append(float(l_df["Accuracy"].values[0]))

data = [fractal_acc, logpower_acc]
labels = ["Fractal", "LogPower"]

plt.figure(figsize=(7, 5))
plt.boxplot(data, tick_labels=labels, patch_artist=True)
plt.ylabel("Acurácia")
plt.title("Boxplot das acurácias - WCCI2020: Fractal vs LogPower")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Salvar figura na pasta graphics/scripts/final_paper_defense/wcci2020/results
output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "boxplot_fractal_vs_logpower.png")
plt.savefig(output_path, dpi=300)
print(f"Figura salva em: {output_path}")
