from datasets import load_dataset
import faiss
import numpy as np
from tqdm import tqdm
import torch

class VectorStoreFAISS:
    def __init__(self, index_path, dataset_name, embedding_model, split='train', device='cuda'):
        self.index = faiss.read_index(index_path)
        self.documents = load_dataset(dataset_name, split=split)
        self.device = device
        self.embedding_model = embedding_model

    def buscar_por_batches(self, dataset_consultas, column, top_k=10, batch_size=1):
        resultados_totales = []
        num_batches = len(dataset_consultas) // batch_size + int(len(dataset_consultas) % batch_size != 0)
        for i in tqdm(range(0, len(dataset_consultas), batch_size), desc="🔍 Buscando"):
            batch = dataset_consultas.select(range(i, min(i + batch_size, len(dataset_consultas))))
            textos_batch = batch[column]

            with torch.no_grad():
                embeddings_batch = self.embedding_model.encode(
                    textos_batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    device=self.device,
                    show_progress_bar=False
                )

            embeddings_batch = np.array(embeddings_batch).astype('float32')
            distancias, indices = self.index.search(embeddings_batch, top_k)

            for j, consulta in enumerate(textos_batch):
                resultados = [
                    (self.documents[int(idx)]['text'], distancias[j][k], self.documents[int(idx)]['file_name'])
                    for k, idx in enumerate(indices[j])
                ]
                resultados_totales.append((consulta, resultados))

        return resultados_totales

    def buscar_una_consulta(self, consulta, top_k=10):
        # Generar embedding para la consulta
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                [consulta],
                normalize_embeddings=True,
                device=self.device,
                show_progress_bar=False
            )

        # Convertir embedding a float32 para FAISS
        embedding = np.array(embedding).astype('float32')

        # Búsqueda en FAISS
        distancias, indices = self.index.search(embedding, top_k)

        # Almacenar resultados
        resultados = [
            (self.documents[int(idx)]['text'], distancias[0][k], self.documents[int(idx)]['file_name'])
            for k, idx in enumerate(indices[0])
        ]

        return consulta, resultados