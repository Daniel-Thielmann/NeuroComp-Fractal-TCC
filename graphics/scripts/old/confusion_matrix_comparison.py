import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_predictions_and_labels(method):
    base_dir = f"results/{method}"
    labels, preds = [], []

    for phase in ["Training", "Evaluate"]:
        folder = os.path.join(base_dir, phase)
        for file in sorted(os.listdir(folder)):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, file))
                df = df[df["true_label"].isin([1, 2])]
                labels.extend(df["true_label"])
                predicted = np.where(df["left_prob"] > df["right_prob"], 1, 2)
                preds.extend(predicted)

    return np.array(labels), np.array(preds)


os.makedirs("graphics/results", exist_ok=True)

labels_h, preds_h = load_predictions_and_labels("Fractal")
labels_l, preds_l = load_predictions_and_labels("LogPower")

cm_h = confusion_matrix(labels_h, preds_h, labels=[1, 2])
cm_l = confusion_matrix(labels_l, preds_l, labels=[1, 2])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(
    cm_h,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Left", "Right"],
    yticklabels=["Left", "Right"],
    ax=axes[0],
)
axes[0].set_title("Fractal")
axes[0].set_xlabel("Predito")
axes[0].set_ylabel("Real")

sns.heatmap(
    cm_l,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Left", "Right"],
    yticklabels=["Left", "Right"],
    ax=axes[1],
)
axes[1].set_title("LogPower")
axes[1].set_xlabel("Predito")
axes[1].set_ylabel("")

plt.suptitle("Matriz de Confusao por Metodo - Fractal vs LogPower", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path = "graphics/results/confusion_matrix_comparison.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico salvo em: {output_path}")
