python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "llm" \
  --reader_model_name "microsoft/phi-2" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/phi_basic_results.csv" \
  --batch_size 100 \
  --llm_batch_size 20
