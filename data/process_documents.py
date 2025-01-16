from llama_index.core import SimpleDirectoryReader
from langchain.docstore.document import Document as LangchainDocument
import pandas as pd
from tqdm.notebook import tqdm
from multiprocessing import cpu_count
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils_data import save_docs_to_jsonl, load_docs_from_jsonl_llama, remove_text_before_second_scope


EMBEDDING_MODEL_NAME = "dinho1597/bge-small-qa-telecom-ft"

def read_documents_from_directory(directory_path):
    num_workers = min(2, cpu_count())
    documents = SimpleDirectoryReader(input_dir=directory_path).load_data(num_workers=num_workers)
    return documents

def create_text_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", ",", ";", " ", ""]
    )

def process_documents(raw_documents, text_splitter):
    processed_docs = []
    for doc in tqdm(raw_documents):
        processed_docs += text_splitter.split_documents([doc])
    return processed_docs

def remove_duplicates(docs):
    unique_texts = {}
    unique_docs = []
    for doc in docs:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            unique_docs.append(doc)
    return unique_docs

def plot_document_lengths(docs, tokenizer, plot_path):
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs)]
    plt.figure()
    pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.savefig(plot_path)
    plt.show()


def main(directory_path, save_path, chunk_size=100, chunk_overlap=20):
    docs_raw_path = save_path+"/raw_documents.jsonl"   
    if os.path.exists(docs_raw_path):
        documents = load_docs_from_jsonl_llama(docs_raw_path)
        print(f"Loaded {len(documents)} docs from JSONL.")
    else:
        documents = read_documents_from_directory(directory_path)
        print("Archivo JSONL no encontrado. Documentos cargados desde el directorio.")
        print(f"Loaded {len(documents)} docs")
        save_docs_to_jsonl(documents, save_path+'/raw_documents.jsonl')
    
    raw_knowledge_base = [
        LangchainDocument(page_content=remove_text_before_second_scope(doc.text), metadata=doc.metadata) for doc in tqdm(documents)
    ]

    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    docs_processed = process_documents(raw_knowledge_base, text_splitter)
    docs_unique = remove_duplicates(docs_processed)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    plot_document_lengths(docs_processed, tokenizer, save_path+f'/document_lengths_{chunk_size}cs_{chunk_overlap}o.png')
    save_docs_to_jsonl(docs_unique, save_path+f'/split_{chunk_size}cs_{chunk_overlap}o.jsonl')
    print(f"Processed documents saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process and split documents with token analysis.")
    parser.add_argument('--directory_path', type=str, required=True, help='Directory path of documents.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the processed JSONL.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for text splitting.')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='Chunk overlap for text splitting.')
    args = parser.parse_args()
    
    main(
        directory_path=args.directory_path,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
