"""
Script temporário para processar apenas o sujeito 10
"""

import os
import pandas as pd
from tqdm import tqdm
from methods.pipelines.csp_fractal import run_csp_fractal
from methods.pipelines.csp_logpower import run_csp_logpower
from methods.pipelines.fbcsp_fractal import run_fbcsp_fractal
from methods.pipelines.fbcsp_logpower import run_fbcsp_logpower

# Sujeito a ser processado
subject_id = 10

for name, func in [
    ("CSP_Fractal", run_csp_fractal),
    ("CSP_LogPower", run_csp_logpower),
    ("FBCSP_Fractal", run_fbcsp_fractal),
    ("FBCSP_LogPower", run_fbcsp_logpower),
]:
    print(f"Processando {name} para o sujeito P{subject_id:02d}...")
    os.makedirs(f"results/{name}/Training", exist_ok=True)
    os.makedirs(f"results/{name}/Evaluate", exist_ok=True)

    rows = func(subject_id)
    df = pd.DataFrame(rows)
    df[df["fold"] < 4].to_csv(
        f"results/{name}/Training/P{subject_id:02d}.csv", index=False
    )
    df[df["fold"] == 4].to_csv(
        f"results/{name}/Evaluate/P{subject_id:02d}.csv", index=False
    )
    print(f"  ✓ Concluído")

print("\nProcessamento do sujeito 10 concluído para todos os métodos!")
