from faiss_vector_store import VectorStoreFAISS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from rag_pipeline import RAGPipeline
from datasets import load_dataset
import pandas as pd

EMBEDDING_MODEL_NAME = "dinho1597/phi-2-telecom-ft"
READER_MODEL_NAME = "microsoft/phi-2"
RERANKER_MODEL_NAME = "colbert-ir/colbertv2.0"
index_path = "/content/drive/MyDrive/Papers/SBRC2025/Faisstest/indice_faiss_cpu.index"
documents_dataset_name = "dinho1597/3GPP-Documents-100cs"
test_dataset_name = "dinho1597/3GPP-QA-MultipleChoice"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
torch.set_default_device(device)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


reader_model = AutoModelForCausalLM.from_pretrained(
    READER_MODEL_NAME,
    torch_dtype="auto",  # Detecta automáticamente el tipo de datos adecuado
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

reranker_model = RAGPretrainedModel.from_pretrained(RERANKER_MODEL_NAME, verbose = 0)

vector_store = VectorStoreFAISS(index_path, documents_dataset_name, embedding_model)

READER_LLM = pipeline(
    model=reader_model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=20,
)

my_rag_pipeline = RAGPipeline(llm=READER_LLM, knowledge_index=vector_store, reranker=reranker_model)

test_dataset = load_dataset('dinho1597/3GPP-QA-MultipleChoice', split='test')

ans = my_rag_pipeline.answer_batch(test_dataset, column='question', batch_size=100)


df = pd.DataFrame({
    'answer': ans[0],
    'valid_options': ans[1]
})

# Guardar el DataFrame en un archivo CSV
ruta_csv = '/mnt/data/answer_test_ll.csv'
df.to_csv(ruta_csv, index=False)