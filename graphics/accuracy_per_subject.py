import os
import pandas as pd
import matplotlib.pyplot as plt

# Garante que a pasta graphics exista
os.makedirs("graphics", exist_ok=True)

# Inicializa listas para armazenar as médias por sujeito
subjects = [f"P{i:02d}" for i in range(1, 11)]
higuchi_means = []
logpower_means = []

# Lê os dados dos CSVs de TRAINING
for i in range(1, 11):
    higuchi_path = f"results/Higuchi/Training/P{i:02d}.csv"
    logpower_path = f"results/LogPower/Training/P{i:02d}.csv"

    hig_df = pd.read_csv(higuchi_path)
    log_df = pd.read_csv(logpower_path)

    # Calcula a média de acerto com base na probabilidade correta (left/right)
    hig_df['correct_prob'] = hig_df.apply(
        lambda row: row['left_prob'] if row['true_label'] == 1 else row['right_prob'], axis=1)
    log_df['correct_prob'] = log_df.apply(
        lambda row: row['left_prob'] if row['true_label'] == 1 else row['right_prob'], axis=1)

    higuchi_means.append(hig_df['correct_prob'].mean())
    logpower_means.append(log_df['correct_prob'].mean())

# Gera o gráfico
plt.figure(figsize=(10, 6))
plt.plot(subjects, higuchi_means, marker='o', label='Higuchi', color='blue')
plt.plot(subjects, logpower_means, marker='s', label='LogPower', color='red')
plt.title("Acurácia Média por Sujeito - Higuchi vs LogPower")
plt.xlabel("Sujeito")
plt.ylabel("Probabilidade média correta")
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Salva o gráfico
plt.savefig("graphics/accuracy_per_subject.png")
plt.close()

print("Gráfico de acurácia por sujeito salvo em graphics/accuracy_per_subject.png")
