import os
import scipy.io


def check_channels_bciciv2b():
    mat_dir = "dataset/BCICIV2b/"
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]
    print(f"Arquivos .mat encontrados: {mat_files}")
    for mat_file in mat_files:
        mat_path = os.path.join(mat_dir, mat_file)
        try:
            mat = scipy.io.loadmat(mat_path)
            if "data" in mat:
                data = mat["data"]
            elif "X" in mat:
                data = mat["X"]
            else:
                print(f"{mat_file}: Nenhuma chave 'data' ou 'X' encontrada.")
                continue
            print(f"{mat_file}: shape {data.shape}")
        except Exception as e:
            print(f"Erro ao ler {mat_file}: {e}")


if __name__ == "__main__":
    check_channels_bciciv2b()
