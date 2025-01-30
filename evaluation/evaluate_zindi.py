import pandas as pd
import ast
from evaluate_answers import load_results, simple_extract_answer

csv_path = './data/testing/all_ft_rerank10_zindi_results.csv'
results = pd.read_csv(
    csv_path,
    converters={
        'question_id': int,
        'inference': ast.literal_eval,
        'valid_options': ast.literal_eval
    }
)
results = results.rename(columns={"question_id": "Question_ID"})

template = pd.read_csv('./data/testing/SampleSubmission.csv')
template = template[['Question_ID']]

ans = []
for index, row in results.iterrows():
    ans.append(simple_extract_answer(row['inference'][0]['generated_text'], row['valid_options']))
results['Answer_ID'] = ans


map_ans={'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
results['Answer_ID']=results['Answer_ID'].map(map_ans)
results['Answer_ID'] = results['Answer_ID'].astype(int)
merged_df = pd.merge(template, results, on='Question_ID')

merged_df = merged_df[['Question_ID', 'Answer_ID']]
merged_df.to_csv('./data/testing/final_zindi.csv', index=False)

merged_df2 = merged_df.copy()
merged_df2['Task'] = "Phi-2"

merged_df2.to_csv('./data/testing/final_zindi2.csv', index=False)
