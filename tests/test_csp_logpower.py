import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods.pipelines.csp_logpower import run_csp_logpower
import pandas as pd

if __name__ == "__main__":
    subject_id = 1
    rows = run_csp_logpower(subject_id)
    df = pd.DataFrame(rows)
    os.makedirs("results/CSP_LogPower/Training", exist_ok=True)
    df.to_csv(f"results/CSP_LogPower/Training/P{subject_id:02d}.csv", index=False)
    print("Teste CSP + LogPower finalizado com sucesso.")
