from datasets import load_dataset

DATASET_NAME = 'dinho1597/Telecom-QA-MultipleChoice'
TEST_DATASET_NAME = 'dinho1597/3GPP-QA-MultipleChoice'

def load_and_prepare_datasets(seed=42):
    # Load Dataset
    dataset = load_dataset(DATASET_NAME, split='train')
    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column='category')
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    train_dataset = train_dataset.rename_columns({
        "question_id": "q_id",
        "question": "anchor",
        "explanation": "positive"
    })
    val_dataset = val_dataset.rename_columns({
        "question_id": "q_id",
        "question": "anchor",
        "explanation": "positive"
    })

    test_dataset = load_dataset(TEST_DATASET_NAME, split='train')
    test_dataset = test_dataset.select_columns(['question_id', 'question', 'explanation'])
    test_dataset = test_dataset.rename_columns({"question_id": "q_id", "question": "anchor", "explanation": "positive"})
    
    return train_dataset, val_dataset, test_dataset

def print_hola():
    print("Hola")