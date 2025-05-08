import pandas as pd
import os


def log_results(method_name: str, base_dir="results"):
    method_dir = os.path.join(base_dir, method_name, "Training")
    if not os.path.exists(method_dir):
        print(f"[LOG] Diretório não encontrado: {method_dir}")
        return

    all_rows = []

    for filename in sorted(os.listdir(method_dir)):
        if filename.endswith(".csv"):
            path = os.path.join(method_dir, filename)
            df = pd.read_csv(path)
            df["correct_prob"] = df.apply(
                lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"], axis=1)
            accuracy = (df["correct_prob"] >= 0.5).mean()
            print(f"{filename}: Acurácia = {accuracy:.4f} | Amostras = {len(df)}")
            all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    overall_accuracy = (df_all["correct_prob"] >= 0.5).mean()
    print(f"\n=== Resultado Geral [{method_name}] ===")
    print(f"Acurácia Total: {overall_accuracy:.4f} ({len(df_all)} amostras)")
    print(
        f"Distribuição de Rótulos: {df_all['true_label'].value_counts().to_dict()}")
