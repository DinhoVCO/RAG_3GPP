import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from sentence_transformers import SentenceTransformer
from data.embedding_train_data import load_and_prepare_datasets
from helpers.ir_evaluator import create_evaluator_information_retrieval



def parse_arguments():
    parser = argparse.ArgumentParser(description="Embedding model training")
    parser.add_argument('--output_dir', type=str, default="/results/embedding/", help="Output directory for the results")
    parser.add_argument('--models_dir', type=str, default="/models/embedding/", help="Directory for the models")
    args = parser.parse_args()
    return args



seed = 42

def load_models(model_dir, model_names):
    print("Cargando modelos...")
    models = {}

    # Recorrer todos los archivos en la carpeta de modelos
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        if os.path.isdir(model_path):  # Verifica que sea un directorio de modelo
            model_key = f"ft_{model_name[17:]}"
            models[model_key] = SentenceTransformer(model_path)

    for model_key, model_name in model_names.items():
        models[model_key] = SentenceTransformer(model_name)

    return models

# Modelos de lenguaje pre-entrenados
model_names = {
    'small': 'BAAI/bge-small-en-v1.5',
    'base': 'BAAI/bge-base-en-v1.5',
    'large': 'BAAI/bge-large-en-v1.5'
}


def main():
    args = parse_arguments()
    output_dir = args.output_dir
    models_dir = args.models_dir

    models = load_models(models_dir, model_names)
    print("Modelos cargados")
    print("Cargando datasets...")
    # Evaluador
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(seed)
    print("Datasets cargados")
    print("Evaluando modelos...")
    # Evaluator
    evaluator = create_evaluator_information_retrieval(test_dataset)
    print("Modelos evaluados")

    print("Guardando métricas...")
    metrics = {}
    for model_key, model_to_evaluate in models.items():
        metrics[model_key] = evaluator(model_to_evaluate)

    # Guardar las métricas en formato JSON
    with open(f"{output_dir}/metrics_embedding.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()