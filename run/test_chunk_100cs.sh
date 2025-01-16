#Evaluar tamaño del chunk size 100, utilizar:
# embedding basico
# reader basico
# comparar los 2 indices faiss de diferentes chunck size
#!/bin/bash
python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "ranker" \
  --reader_model_name "microsoft/phi-2" \
  --embedding_model_name "BAAI/bge-small-en-v1.5" \
  --index_path "/content/drive/MyDrive/Papers/RAG_3GPP/index/faiss_cpu_100cs.index" \
  --documents_dataset_name "dinho1597/3GPP-docs-100cs" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/chunk_100_results.csv" \
  --batch_size 80 \
  --llm_batch_size 10 \
  --num_retriever_docs 3
