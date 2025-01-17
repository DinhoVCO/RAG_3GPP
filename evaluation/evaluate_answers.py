import pandas as pd
import random
import ast
from collections import Counter
from datasets import load_dataset
import os

def load_results(csv_path, test_dataset):
    test_dataset = load_dataset(test_dataset, split ='test')
    test_df = pd.DataFrame(test_dataset)
    test_df = test_df[['question_id','category']]
    result = pd.read_csv(
        csv_path,
        converters={
            'question_id': int,
            'inference': ast.literal_eval,
            'valid_options': ast.literal_eval
        }
    )
    merged_df = pd.merge(result, test_df, on='question_id')
    return merged_df

def simple_extract_answer(model_output, valid_options):
    values = [opt + ')' for opt in valid_options]
    lines = model_output.splitlines()

    for line in lines:
        texto_sin_puntos = line.replace('.', '')
        words = texto_sin_puntos.strip().split()
        matches = {
            'with_paren': [word for word in words if word in values],
            'without_paren': [word for word in words if word in valid_options]
        }

        for key in matches:
            if matches[key]:
                most_common, _ = Counter(matches[key]).most_common(1)[0]
                matches[key] = most_common
            else:
                matches[key] = None

        if matches['with_paren'] and matches['without_paren']:
            return matches['without_paren'] if matches['with_paren'][0] == matches['without_paren'] else matches['with_paren'][0]
        elif matches['with_paren']:
            return matches['with_paren'][0]
        elif matches['without_paren']:
            return matches['without_paren']

    values2 = ['(' + opt + ')' for opt in valid_options]
    for line in lines:
        words = line.strip().split()
        matches = [word.upper() for word in words if word.upper() in values2]
        if matches:
            return matches[0][1]

    print('------------')
    print('retornando valor aleatorio')
    print(model_output)
    print('------------')
    return random.choice(valid_options)

def evaluate_results(test,results,path_save):
    ans = []
    for index, row in results.iterrows():
        ans.append(simple_extract_answer(row['inference'][0]['generated_text'], row['valid_options']))

    results['respuesta_usuario'] = ans
    results['correcta'] = results['answer'] == results['respuesta_usuario']
    precision = results['correcta'].mean() * 100

    if not os.path.exists(path_save+'/accuracy.txt'):
        with open(path_save+'/accuracy.txt', 'w') as file:
            file.write('test, categoria ,precision\n') 
    with open(path_save+'/accuracy.txt', 'a') as file:
        precision = results['correcta'].mean() * 100
        file.write(f'{test},-,{precision}\n')  
        categories = results['category'].unique()
        for category in categories:
            category_df = results[results['category'] == category]
            precision_c = category_df['correcta'].mean() * 100
            file.write(f'{test},{category},{precision_c}\n')  

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluar precisión de respuestas extraídas.")
    parser.add_argument('--results_path', type=str, required=True, help="Ruta al archivo CSV de resultados.")
    parser.add_argument('--test_dataset', type=str, required=True, help="Nombre del dataset de test.")
    parser.add_argument('--save_path', type=str, required=True, help="Direccion opara guardar las evaluacion")

    args = parser.parse_args()
    nombre_archivo = os.path.basename(args.results_path)
    results = load_results(args.results_path, args.test_dataset)
    evaluate_results(nombre_archivo,results,args.save_path)


if __name__ == "__main__":
    main()
