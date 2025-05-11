import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from methods.pipelines.csp_logpower import run_csp_logpower


def run_all():
    all_rows = []

    for subject_id in tqdm(range(1, 10), desc="CSP_Logpower"):
        rows = run_csp_logpower(subject_id)
        df = pd.DataFrame(rows)
        os.makedirs("results/CSP_LogPower/Training", exist_ok=True)
        df.to_csv(f"results/CSP_LogPower/Training/P{subject_id:02d}.csv", index=False)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    df = run_all()
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )
    acc = (df["true_label"] == df["left_prob"].lt(0.5).astype(int) + 1).mean()
    mean_prob = df["correct_prob"].mean()
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())

    print(
        f"Acurácia: {acc:.4f} | Média Prob. Correta: {mean_prob:.4f} | Amostras: {total} | Rótulos: {counts}"
    )
