import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Parâmetros do sinal sintético
n_trials = 10
n_channels = 1
sampling_rate = 160  # Hz
duration_secs = 4
n_samples = sampling_rate * duration_secs  # 640

# Caminho de saída
SAVE_PATH = "graphics/visualizations/eeg_animation.gif"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Gerar dados sintéticos: [n_trials, n_channels, n_samples]
np.random.seed(42)
data = np.random.randn(n_trials, n_channels, n_samples) * 10

# Configurar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f"Canal {i+1}")[0] for i in range(n_channels)]
ax.set_xlim(0, n_samples)
ax.set_ylim(-30, 30)
ax.set_xlabel("Tempo (amostras)")
ax.set_ylabel("Amplitude EEG")
ax.set_title("Evolução dos Sinais EEG ao Longo do Tempo")
ax.legend(loc="upper right")


# Inicializar as linhas
def init():
    for line in lines:
        line.set_data([], [])
    return lines


# Atualizar por trial
def update(frame):
    ax.set_title(f"Trial {frame+1} - Visualizando {n_channels} canais EEG")
    for i, line in enumerate(lines):
        line.set_data(np.arange(n_samples), data[frame, i])
    return lines


# Criar animação
ani = FuncAnimation(
    fig, update, frames=n_trials, init_func=init, blit=True, interval=1000, repeat=False
)

# Salvar como GIF
ani.save(SAVE_PATH, writer=PillowWriter(fps=1))
print(f"GIF salvo em: {SAVE_PATH}")
