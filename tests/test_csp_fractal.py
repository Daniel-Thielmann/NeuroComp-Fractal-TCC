import pandas as pd
from methods.pipelines.csp_fractal import run_csp_fractal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    subject_id = 1
    rows = run_csp_fractal(subject_id=subject_id)

    df = pd.DataFrame(rows)
    os.makedirs("results/CSP_Fractal/Training", exist_ok=True)
    df.to_csv(
        f"results/CSP_Fractal/Training/P{subject_id:02d}.csv", index=False)

    print("Teste CSP + Fractal finalizado com sucesso.")
