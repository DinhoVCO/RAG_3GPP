from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import json
import numpy as np
from huggingface_hub import HfApi
from huggingface_hub import login

login(token="")
api = HfApi()

file_path1 = "./RAG_3GPP/data/testing/TeleQnA_testing1.txt"
file_path2 = "./RAG_3GPP/data/testing/questions_new.txt"

with open(file_path1, "r") as f:
    test_data = f.read()

with open(file_path2, "r") as f:
    test2_data = f.read()

test_data=json.loads(test_data)
test2_data = json.loads(test2_data)

combined_test_data = {**test_data, **test2_data}

def convert_dict_to_dataframe(data):
    """
    Converts a dictionary into a pandas DataFrame with specific columns.

    Args:
        data (dict): Dictionary where each key is a question ID, and the value is another dictionary
                     containing question information and options.

    Returns:
        pd.DataFrame: DataFrame with structured columns.
    """
    df_data = []
    for key, value in data.items():
        # Extracting the answer option (e.g., "option 2")
        #answer_option = value.get("answer", "").split(":")[0].strip().lower()
        # Mapping answer option to A, B, C, D, E
        answer_mapping = {
            "option 1": "A",
            "option 2": "B",
            "option 3": "C",
            "option 4": "D",
            "option 5": "E"
        }
        row = {
            "question_id": key,
            "question": value.get("question"),
            "A": value.get("option 1"),
            "B": value.get("option 2"),
            "C": value.get("option 3", None),
            "D": value.get("option 4", None),
            "E": value.get("option 5", None),
            "answer":  None,
            "explanation": value.get("explanation", None),
            "category": value.get("category", None),
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df['question'] = df['question'].str.replace(r'\s*\[.*\]$', '', regex=True)
    df['question_id'] = df['question_id'].str.replace('question ', '').astype(int)
    return df

test_df=convert_dict_to_dataframe(combined_test_data)
print(test_df.head())

test_3gpp_dataset = Dataset.from_pandas(test_df, preserve_index=False)
dataset_dict = DatasetDict({"test": test_3gpp_dataset})

# Crear un repositorio en Hugging Face
repo_name = "zindi_test"  # Cambia el nombre del dataset
username = "dinho1597"           # Tu usuario de Hugging Face

# Crear el repositorio (si no existe)
#api.create_repo(repo_id=f"{username}/{repo_name}", repo_type="dataset", exist_ok=True, private=False)
dataset_dict.push_to_hub(f"{username}/{repo_name}")