from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from data.embedding_train_data import load_and_prepare_datasets
from helpers.ir_evaluator import evaluate_information_retrieval
import json
import os

seed = 42
output_dir = "results/embeding"
model_name = "bge-small-telecom_5e_128bs"
model_path = "/models/embedding/"+model_name
#Load my model
model = SentenceTransformer(model_path)

# Modelos de lenguaje pre-entrenados
model_names = {
    'ft_small': None,  # Placeholder para el modelo ya cargado
    'small': 'BAAI/bge-small-en-v1.5',
    'base': 'BAAI/bge-base-en-v1.5',
    'large': 'BAAI/bge-large-en-v1.5'
}

models = {}
for model_key, model_name in model_names.items():
    if model_key == 'FT_small':
        models[model_key] = model  # Asignar el modelo ya cargado
    else:
        models[model_key] = SentenceTransformer(model_name)


# Evaluador
train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(seed)

# Evaluator
evaluator = evaluate_information_retrieval(test_dataset)

metrics = {}
for model_key, model_to_evaluate in models.items():
  metrics[model_key] = evaluator(model_to_evaluate)

# Guardar las métricas en formato JSON
with open(f"{output_dir}/metrics_{model_name}.json", "w") as f:
    json.dump(metrics, f, indent=4)