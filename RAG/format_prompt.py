from string import Template
import pandas as pd

template_RAG = Template(
    "Instruct: Use the context provided to select the correct option. Select the correct option from $valid_options. Respond with the letter of the correct option. (e.g., 'A').\n"
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
def format_input_context(row, context=None):
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
        # Usar el template con explicación
        input_text = template_RAG.substitute(
            valid_options=valid_options_text,
            question=question_text,
            options=options_text,
            explanation=context
        )
    else:
        # Usar el template sin explicación
        input_text = template_base.substitute(
            valid_options=valid_options_text,
            question=question_text,
            options=options_text
        )

    return input_text, valid_options