import os
import mne
import scipy.io as sio

input_folder = "dataset/BCICIV2a"

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

            # Extrai os sinais
            signals = raw.get_data()

            # Extrai eventos e rótulos
            events, _ = mne.events_from_annotations(raw, verbose=False)
            labels = []
            for desc in raw.annotations.description:
                if desc.startswith("769"):
                    labels.append(1)  # left hand
                elif desc.startswith("770"):
                    labels.append(2)  # right hand
                elif desc.startswith("771"):
                    labels.append(3)  # foot
                elif desc.startswith("772"):
                    labels.append(4)  # tongue
                else:
                    labels.append(0)  # outros

            # Constrói o dicionário para salvar
            mat_data = {
                "signals": signals,
                "sfreq": raw.info["sfreq"],
                "ch_names": raw.ch_names,
                "events": events,
                "labels": labels,
            }

            mat_filename = subject_name + ".mat"
            mat_path = os.path.join(input_folder, mat_filename)
            sio.savemat(mat_path, mat_data)

            print(f"✔️  {mat_filename} salvo com sucesso.")
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
