python /content/RAG_3GPP/RAG/main_rag.py \
  --inference_type "llm" \
  --reader_model_name "dinho1597/phi-2-telecom-ft-v1_test" \
  --test_dataset_name "dinho1597/3GPP-QA-MultipleChoice" \
  --output_csv_path "/content/drive/MyDrive/Papers/RAG_3GPP/results/phi_ft_v1_results.csv" \
  --batch_size 100 \
  --llm_batch_size 20
