import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import torch

def generar_embeddings_batch_gpu(modelo, dataset, batch_size, dispositivo):
    """
    Genera embeddings directamente en la GPU usando procesamiento por batches.
    """
    embeddings_totales = []
    num_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generando embeddings", total=num_batches):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        textos_batch = batch['text']

        with torch.no_grad():
            embeddings = modelo.encode(
                textos_batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                device=dispositivo,
                show_progress_bar=False
            )
        embeddings_totales.append(embeddings)

    return np.vstack(embeddings_totales)

def crear_indice_faiss(embeddings, path_indice_faiss, dataset_name):
    """
    Crea y guarda un índice FAISS optimizado para la GPU.
    """
    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings.astype('float32'))
    faiss.write_index(indice, f"{path_indice_faiss}/faiss_cpu_{dataset_name[-5:]}.index")
    print("Índice guardado exitosamente.")

def main(nombre_dataset, path_indice_faiss, batch_size=4096):
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {dispositivo}")

    modelo = SentenceTransformer('dinho1597/bge-small-qa-telecom-ft', device=dispositivo)

    dataset = load_dataset(nombre_dataset, split='train')
    dataset = dataset.select_columns(['text', 'file_name'])

    print("Generando embeddings...")
    embeddings = generar_embeddings_batch_gpu(modelo, dataset, batch_size, dispositivo)

    print("Creando índice FAISS...")
    crear_indice_faiss(embeddings, path_indice_faiss, nombre_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar embeddings y crear índice FAISS.")
    parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset.")
    parser.add_argument("--output", type=str, required=True, help="Ruta para guardar el índice FAISS.")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")

    args = parser.parse_args()
    main(args.dataset, args.output)
