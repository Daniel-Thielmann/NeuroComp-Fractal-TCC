import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods.pipelines.fbcsp_fractal import run_fbcsp_fractal
import pandas as pd

if __name__ == "__main__":
    subject_id = 1
    rows = run_fbcsp_fractal(subject_id)
    df = pd.DataFrame(rows)
    os.makedirs("results/FBCSP_Fractal/Training", exist_ok=True)
    df.to_csv(f"results/FBCSP_Fractal/Training/P{subject_id:02d}.csv", index=False)
    print("Teste FBCSP + Fractal finalizado com sucesso.")
