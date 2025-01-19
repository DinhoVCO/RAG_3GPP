python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "reranker" \
  --reader_model_name "dinho1597/phi-2-telecom-ft-v1_test" \
  --embedding_model_name "dinho1597/bge-small-qa-telecom-ft" \
  --index_path "/content/drive/MyDrive/Papers/RAG_3GPP/index/faiss_cpu_100cs.index" \
  --documents_dataset_name "dinho1597/3GPP-docs-100cs" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/all_ft_rerank3_100cs_results.csv" \
  --batch_size 80 \
  --llm_batch_size 10 \
  --num_retriever_docs 10 \
  --num_docs_final 3 \