import pandas as pd
import random
import ast
from collections import Counter

def load_results(csv_path):
    return pd.read_csv(
        csv_path,
        converters={
            'question_id': int,
            'inference': ast.literal_eval,
            'valid_options': ast.literal_eval
        }
    )

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

def evaluate_results(results):
    ans = []
    for index, row in results.iterrows():
        ans.append(simple_extract_answer(row['inference'][0]['generated_text'], row['valid_options']))

    results['respuesta_usuario'] = ans
    results['correcta'] = results['answer'] == results['respuesta_usuario']
    precision = results['correcta'].mean() * 100
    return precision

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluar precisión de respuestas extraídas.")
    parser.add_argument('--results_path', type=str, required=True, help="Ruta al archivo CSV de resultados.")
    args = parser.parse_args()

    results = load_results(args.results_path)
    precision = evaluate_results(results)
    print(f"La precisión es: {precision:.2f}%")

if __name__ == "__main__":
    main()
