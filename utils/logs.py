import pandas as pd
import os


def log_summary(method_name: str, base_dir="results"):
    method_dir = os.path.join(base_dir, method_name, "Training")
    if not os.path.exists(method_dir):
        return f"[{method_name}] → Diretório não encontrado."

    all_rows = []
    for filename in sorted(os.listdir(method_dir)):
        if filename.endswith(".csv"):
            path = os.path.join(method_dir, filename)
            df = pd.read_csv(path)
            df["correct_prob"] = df.apply(
                lambda row: (
                    row["left_prob"] if row["true_label"] == 1 else row["right_prob"]
                ),
                axis=1,
            )
            all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    acc = (df_all["correct_prob"] >= 0.5).mean()
    avg_prob = df_all["correct_prob"].mean()
    count = len(df_all)
    label_dist = df_all["true_label"].value_counts().to_dict()

    return f"[{method_name}] Acurácia: {acc:.4f} | Média Prob. Correta: {avg_prob:.4f} | Amostras: {count} | Rótulos: {label_dist}"
