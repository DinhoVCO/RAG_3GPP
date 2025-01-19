python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "abbre" \
  --reader_model_name "dinho1597/phi-2-telecom-ft-v1_test" \
  --embedding_model_name "dinho1597/bge-small-qa-telecom-ft" \
  --index_path "/content/drive/MyDrive/Papers/RAG_3GPP/index/faiss_cpu_100cs.index" \
  --index_abbre_path "/content/drive/MyDrive/Papers/RAG_3GPP/index/abbreviations_index/faiss_cpu_taset.index" \
  --documents_dataset_name "dinho1597/3GPP-docs-100cs" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/all_ft_abbre_rerank10_100cs_results.csv" \
  --batch_size 80 \
  --llm_batch_size 10 \
  --num_retriever_docs 10 \
  --num_docs_final 10 \
