import os
import sys
from string import Template
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

template_RAG = Template(
    "Instruct: Use the context provided to select the correct option. Select the correct option from $valid_options. Respond with the letter of the correct option. (e.g., 'A').\n"
    "Context:\n$explanation\n\n"
    "Question:\n$question\n\n"
    "Options:\n$options\n\n"
    "Output:"
)

template_RAG_Abbr = Template(
    "Instruct: Use the Abbreviations and Context provided to select the correct option. Select the correct option from $valid_options. Respond with the letter of the correct option. (e.g., 'A').\n"
    "Abbreviations:\n$abbreviations\n\n"
    "Context:\n$explanation\n\n"
    "Question:\n$question\n\n"
    "Options:\n$options\n\n"
    "Output:"
)

template_base = Template(
    "Instruct: Answer the following question by selecting the correct option. Select the correct option from $valid_options. Respond with the letter of the correct option. (e.g., 'A').\n"
    "Question:\n$question\n\n"
    "Options:\n$options\n\n"
    "Output:"
)
def format_input_context(row, context=None, abbreviations=None):
    question_text = row['question']  # Accede directamente a la pregunta en la fila

    # Crear una lista de opciones válidas (diferentes de None)
    options_dict = {
        'A': row['A'],
        'B': row['B'],
        'C': row['C'],
        'D': row['D'],
        'E': row['E'],
    }

    valid_options = [key for key, value in options_dict.items() if pd.notna(value) and value is not None]
    valid_options_text = ", ".join(valid_options)  # Crear una lista dinámica de opciones disponibles

    # Crear el texto de opciones válidas
    options_text = "\n".join([f"{key}) {value}" for key, value in options_dict.items() if key in valid_options])

    if context:
        if abbreviations:
            input_text = template_RAG_Abbr.substitute(
                valid_options=valid_options_text,
                abbreviations=abbreviations,
                question=question_text,
                options=options_text,
                explanation=context
            )
        else:
            input_text = template_RAG.substitute(
                valid_options=valid_options_text,
                question=question_text,
                options=options_text,
                explanation=context
            )
    else:
        input_text = template_base.substitute(
            valid_options=valid_options_text,
            question=question_text,
            options=options_text
        )

    return input_text, valid_options

def get_answer(row):
    ans = row['answer']
    full_ans = row[ans]
    return f"{ans}) {full_ans}"

def get_context(row, include_explanation = False):
    relevant_docs = row['relevant_documents']
    if include_explanation:
        posicion_aleatoria = random.randint(0, len(relevant_docs))
        # Insertar el elemento en la posición aleatoria
        relevant_docs.insert(posicion_aleatoria, row['explanation'])
        #relevant_docs.append(row['explanation'])
    context = ""
    context += "".join(
        [f"\nDocument {str(i)}:" + doc for i, doc in enumerate(relevant_docs)]
    )
    return context

def get_full_promt(row, include_explanation = False):
    context = get_context(row,include_explanation)
    question = format_input_context(row, context)[0]
    answer = get_answer(row)
    return f"{question}\n{answer}"