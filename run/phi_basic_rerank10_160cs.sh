python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "reranker" \
  --reader_model_name "microsoft/phi-2" \
  --embedding_model_name "BAAI/bge-small-en-v1.5" \
  --index_path "/content/drive/MyDrive/Papers/RAG_3GPP/index/faiss_cpu_160cs.index" \
  --documents_dataset_name "dinho1597/3GPP-docs-160cs" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/phi_basic_rerank10_160cs_results.csv" \
  --batch_size 80 \
  --llm_batch_size 10 \
  --num_retriever_docs 20 \
  --num_docs_final 10 \
