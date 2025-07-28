import matplotlib.pyplot as plt
import pandas as pd
import os

base_path = r"d:\dev\EEG-TCC\results\bciciv2a"

csp_fractal_eval_path = os.path.join(base_path, "csp_fractal", "evaluate")
csp_fractal_train_path = os.path.join(base_path, "csp_fractal", "training")
csp_logpower_eval_path = os.path.join(base_path, "csp_logpower", "evaluate")
csp_logpower_train_path = os.path.join(base_path, "csp_logpower", "training")

subjects_eval = [f"P0{i}_evaluate.csv" for i in range(1, 10)]
subjects_train = [f"P0{i}_training.csv" for i in range(1, 10)]

fractal_acc = []
logpower_acc = []

for subj in subjects_eval:
    f_file = os.path.join(csp_fractal_eval_path, subj)
    l_file = os.path.join(csp_logpower_eval_path, subj)
    if os.path.exists(f_file) and os.path.exists(l_file):
        f_df = pd.read_csv(f_file)
        l_df = pd.read_csv(l_file)
        fractal_acc.extend([float(x) for x in f_df["Accuracy"].dropna()])
        logpower_acc.extend([float(x) for x in l_df["Accuracy"].dropna()])

for subj in subjects_train:
    f_file = os.path.join(csp_fractal_train_path, subj)
    l_file = os.path.join(csp_logpower_train_path, subj)
    if os.path.exists(f_file) and os.path.exists(l_file):
        f_df = pd.read_csv(f_file)
        l_df = pd.read_csv(l_file)
        fractal_acc.extend([float(x) for x in f_df["Accuracy"].dropna()])
        logpower_acc.extend([float(x) for x in l_df["Accuracy"].dropna()])

data = [fractal_acc, logpower_acc]
labels = ["CSP + Fractal", "CSP + LogPower"]

plt.figure(figsize=(7, 5))
plt.boxplot(data, tick_labels=labels, patch_artist=True)
plt.ylabel("Acurácia")
plt.title("Boxplot das acurácias - BCICIV2a: CSP + Fractal vs CSP + LogPower")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "boxplot_CSP_fractal_vs_logpower.png")
plt.savefig(output_path, dpi=300)
print(f"Figura salva em: {output_path}")
