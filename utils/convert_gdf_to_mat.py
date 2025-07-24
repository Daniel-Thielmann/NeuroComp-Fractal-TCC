import os
import mne
import scipy.io as sio
import warnings
import numpy as np

# Silenciar warnings específicos do MNE sobre filtros inconsistentes
warnings.filterwarnings(
    "ignore", message=".*Highpass cutoff frequency.*is greater than lowpass.*"
)

input_folder = "dataset/BCICIV2b"

# Cria a pasta se não existir
os.makedirs(input_folder, exist_ok=True)

# Percorre os arquivos .gdf
for filename in os.listdir(input_folder):
    if filename.endswith(".gdf"):
        subject_name = filename.replace(".gdf", "")
        gdf_path = os.path.join(input_folder, filename)
        print(f"Processando: {filename}")

        try:
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
            print(f"[DEBUG] {filename}: raw.ch_names = {raw.ch_names}")

            # Para BCICIV2b, apenas 3 canais EEG: EEG:C3, EEG:Cz, EEG:C4
            eeg_ch_names = ["EEG:C3", "EEG:Cz", "EEG:C4"]
            available_chs = [ch for ch in eeg_ch_names if ch in raw.ch_names]
            if len(available_chs) < 3:
                raise ValueError(
                    f"Arquivo {filename} não possui todos os 3 canais EEG padrão. Encontrados: {available_chs}"
                )
            raw.pick(eeg_ch_names)
            signals = raw.get_data()
            print(f"[DEBUG] {filename}: signals.shape={signals.shape}")

            # Extrai eventos e rótulos
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            print(
                f"[DEBUG] {filename}: eventos brutos extraídos do MNE (shape={events.shape}):\n{events}"
            )
            print(f"[DEBUG] {filename}: event_id mapeado pelo MNE: {event_id}")
            # Eventos de interesse: 769 (left), 770 (right)

            # Filtra apenas eventos de interesse: 10 (left), 11 (right)
            events_bciciv2b = []
            labels = []
            for ev in events:
                code = int(ev[2])
                if code == 10:
                    events_bciciv2b.append([ev[0], 0, 1])
                    labels.append(1)
                elif code == 11:
                    events_bciciv2b.append([ev[0], 0, 2])
                    labels.append(2)
            events_bciciv2b = np.array(events_bciciv2b)
            labels = np.array(labels)

            mat_data = {
                "signals": signals,
                "sfreq": raw.info["sfreq"],
                "ch_names": eeg_ch_names,
                "events": events_bciciv2b,
                "labels": labels,
            }

            mat_filename = subject_name + ".mat"
            mat_path = os.path.join(input_folder, mat_filename)
            sio.savemat(mat_path, mat_data)

            print(f"[OK] {mat_filename} salvo com sucesso.")
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
