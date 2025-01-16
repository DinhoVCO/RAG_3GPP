from typing import Iterable
from langchain.docstore.document import Document as LangchainDocument
from llama_index.core import Document
import json
import re


def save_docs_to_jsonl(array:Iterable[LangchainDocument], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[LangchainDocument]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = LangchainDocument(**data)
            array.append(obj)
    return array

def load_docs_from_jsonl_llama(file_path)->Iterable[LangchainDocument]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def remove_text_before_second_scope(text):
    matches = [m.start() for m in re.finditer("Scope", text)]
    if len(matches) < 2:
        return text
    second_scope_index = matches[1]
    textFiletered = text[second_scope_index:]
    return textFiletered