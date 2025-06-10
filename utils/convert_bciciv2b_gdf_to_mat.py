import os
import mne
import scipy.io as sio
import numpy as np

input_folder = "dataset/BCICIV2b"

# Cria a pasta se não existir
os.makedirs(input_folder, exist_ok=True)

print("Convertendo arquivos BCICIV2b .gdf para .mat...")

# Percorre os arquivos .gdf
for filename in os.listdir(input_folder):
    if filename.endswith(".gdf"):
        subject_name = filename.replace(".gdf", "")
        gdf_path = os.path.join(input_folder, filename)
        print(f"Processando: {filename}")

        try:
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

            # Extrai os sinais
            data = raw.get_data()  # Shape: (channels, samples)

            # Extrai eventos e anotações
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            print(f"  Data shape: {data.shape}")
            print(f"  Events shape: {events.shape}")
            print(f"  Event IDs: {event_id}")

            # Event mapping for BCICIV2b
            event_mapping = {
                769: 1,  # left-hand
                770: 2,  # right-hand
            }

            # Extract trials based on events
            trial_events = []
            trial_labels = []

            for event in events:
                event_time, _, event_code = event
                if event_code in event_mapping:
                    trial_events.append(event_time)
                    trial_labels.append(event_mapping[event_code])

            print(f"  Found {len(trial_events)} valid trials")

            if len(trial_events) == 0:
                print(f"  No valid trials found in {filename}")
                continue

            # Extract trial data (4 seconds after each cue)
            sfreq = raw.info["sfreq"]
            trial_length = int(4 * sfreq)  # 4 seconds
            n_channels = data.shape[0]
            trials = []
            valid_labels = []

            for i, (event_time, label) in enumerate(zip(trial_events, trial_labels)):
                start_sample = int(event_time)
                end_sample = start_sample + trial_length

                if end_sample <= data.shape[1]:
                    trial_data = data[:, start_sample:end_sample]
                    trials.append(trial_data)
                    valid_labels.append(label)

            if len(trials) > 0:
                # Convert to numpy array (trials, channels, samples)
                trials_array = np.stack(trials, axis=0)

                print(f"  Extracted {len(trials)} trials: {trials_array.shape}")

                # Constrói o dicionário para salvar
                mat_data = {
                    "RawEEGData": trials_array,
                    "Labels": np.array(valid_labels),
                    "sfreq": sfreq,
                    "ch_names": raw.ch_names,
                    "events": events,
                }

                mat_filename = (
                    "parsed_P" + subject_name[1:] + ".mat"
                )  # B0101T -> parsed_P0101T.mat
                mat_path = os.path.join(input_folder, mat_filename)
                sio.savemat(mat_path, mat_data)

                print(f"✔️  {mat_filename} salvo com sucesso.")
            else:
                print(f"  No trials extracted from {filename}")

        except Exception as e:
            print(f"❌ Erro ao processar {filename}: {e}")

print("Conversão concluída!")
