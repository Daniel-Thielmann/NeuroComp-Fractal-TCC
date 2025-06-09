import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define methods to analyze
methods = [
    "Fractal",
    "LogPower",
    "CSP_Fractal",
    "CSP_LogPower",
    "FBCSP_Fractal",
    "FBCSP_LogPower",
]


# Function to calculate kappa for a subject and method
def calculate_kappa(subject_id, method):
    kappa = None
    try:
        # Collect all predictions and true labels
        true_labels = []
        predictions = []

        # Process both Training and Evaluate datasets
        for subset in ["Training", "Evaluate"]:
            file_path = f"results/{method}/{subset}/P{subject_id:02d}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df[df["true_label"].isin([1, 2])]  # Keep only classes 1 and 2

                # Calculate predictions based on probabilities
                pred_labels = (df["left_prob"] < 0.5).astype(int) + 1

                true_labels.extend(df["true_label"].values)
                predictions.extend(pred_labels)

        if true_labels and predictions:
            kappa = cohen_kappa_score(true_labels, predictions)
    except Exception as e:
        print(f"Error calculating kappa for subject {subject_id}, method {method}: {e}")

    return kappa


# Create a DataFrame to store results
results = pd.DataFrame(columns=["Subject"] + methods)

# Calculate kappa for each subject and method
for subject_id in range(1, 10):  # Subjects 1-9
    row = {"Subject": f"P{subject_id:02d}"}

    for method in methods:
        kappa = calculate_kappa(subject_id, method)
        row[method] = kappa

    new_row = pd.DataFrame([row])
    if not new_row.isnull().all(axis=1).iloc[0]:
        # Check if the results DataFrame is empty
        if results.empty:
            results = new_row.copy()
        else:
            # Create a copy of the result before concatenation
            results_copy = results.copy()
            # Concatenate with copy to avoid warnings
            results = pd.concat([results_copy, new_row], ignore_index=True)
# Add mean row
mean_row = {"Subject": "Mean"}
for method in methods:
    mean_row[method] = results[method].mean()

# Create DataFrame with the mean row
mean_df = pd.DataFrame([mean_row])

# Check if results is empty before concatenation
if results.empty:
    results = mean_df.copy()
else:
    # Create a copy of the result before concatenation
    results_copy = results.copy()
    # Concatenate with copy to avoid warnings
    results = pd.concat([results_copy, mean_df], ignore_index=True)

# Format and display the table
formatted_results = results.copy()
for col in methods:
    formatted_results[col] = formatted_results[col].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
    )

# Save the table to CSV (table will be printed by main.py)
results.to_csv("results/summaries/kappa_by_subject_method.csv", index=False)

# Create a heatmap visualization
plt.figure(figsize=(12, 8))
numeric_results = results.iloc[:-1].copy()  # Exclude mean row
numeric_results = numeric_results.set_index("Subject")

# Convert to numeric (needed in case some values are NaN or strings)
numeric_data = numeric_results.apply(pd.to_numeric, errors="coerce")

im = plt.imshow(numeric_data.values, cmap="viridis")
plt.colorbar(im, label="Cohen's Kappa")

# Set labels
plt.xticks(np.arange(len(methods)), methods, rotation=45, ha="right")
plt.yticks(np.arange(len(numeric_data.index)), numeric_data.index)

# Add text annotations
for i in range(len(numeric_data.index)):
    for j in range(len(methods)):
        value = numeric_data.iloc[i, j]
        if pd.notnull(value):
            text_color = "white" if value < 0.5 else "black"
            plt.text(j, i, f"{value:.4f}", ha="center", va="center", color=text_color)
        else:
            plt.text(j, i, "N/A", ha="center", va="center", color="white")

plt.tight_layout()
plt.title("Cohen's Kappa by Subject and Method")
plt.savefig("graphics/results/kappa_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
