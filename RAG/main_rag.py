from faiss_vector_store import VectorStoreFAISS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from rag_pipeline import RAGPipeline
from datasets import load_dataset
import pandas as pd
import argparse

def load_embedding_model(model_name, device):
    return SentenceTransformer(model_name, device=device)

def load_reader_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_reranker_model(model_name):
    return RAGPretrainedModel.from_pretrained(model_name, verbose=0)

def initialize_vector_store(index_path, dataset_name, embedding_model):
    return VectorStoreFAISS(index_path, dataset_name, embedding_model)

def create_rag_pipeline(reader_model, tokenizer, vector_store, reranker_model):
    reader_llm = pipeline(
        model=reader_model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=20,
    )
    return RAGPipeline(llm=reader_llm, knowledge_index=vector_store, reranker=reranker_model)

def load_test_dataset(dataset_name):
    return load_dataset(dataset_name, split='test')

def save_answers_to_csv(answers, valid_options, file_path):
    df = pd.DataFrame({
        'answer': answers,
        'valid_options': valid_options
    })
    df.to_csv(file_path, index=False)

def main(inference_type, embedding_model_name, reader_model_name, reranker_model_name, index_path, documents_dataset_name, test_dataset_name, output_csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    torch.set_default_device(device)

    embedding_model = None
    reranker_model = None
    vector_store = None
    reader_model, tokenizer = load_reader_model(reader_model_name, device)
    if(inference_type == 'ranker' ):
        embedding_model = load_embedding_model(embedding_model_name, device)
        vector_store = initialize_vector_store(index_path, documents_dataset_name, embedding_model)
    if(inference_type == 'reranker'):
        embedding_model = load_embedding_model(embedding_model_name, device)
        reranker_model = load_reranker_model(reranker_model_name)
        vector_store = initialize_vector_store(index_path, documents_dataset_name, embedding_model)
    rag_pipeline = create_rag_pipeline(reader_model, tokenizer, vector_store, reranker_model)

    test_dataset = load_test_dataset(test_dataset_name)
    answers, valid_options = rag_pipeline.answer_batch(test_dataset, column='question', batch_size=100)

    save_answers_to_csv(answers, valid_options, output_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Execution")
    parser.add_argument("--inference_type", type=str, default='llm', help="Type of inference to perform (llm, ranker, or reranker)")
    parser.add_argument("--embedding_model_name", type=str, default='BAAI/bge-small-en-v1.5', help="Name of the embedding model")
    parser.add_argument("--reader_model_name", type=str, default='microsoft/phi-2', help="Name of the reader model")
    parser.add_argument("--reranker_model_name", type=str, default='colbert-ir/colbertv2.0', help="Name of the reranker model")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the FAISS index")
    parser.add_argument("--documents_dataset_name", type=str, required=True, help="Name of the documents dataset")
    parser.add_argument("--test_dataset_name", type=str, required=True, help="Name of the test dataset")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the answer output CSV")

    args = parser.parse_args()

    main(
        args.input_type,
        args.embedding_model_name,
        args.reader_model_name,
        args.reranker_model_name,
        args.index_path,
        args.documents_dataset_name,
        args.test_dataset_name,
        args.output_csv_path
    )
