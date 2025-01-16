import pandas as pd
from typing import Optional, List, Tuple
from faiss_vector_store import VectorStoreFAISS
from transformers import pipeline
from ragatouille import RAGPretrainedModel
from tqdm import tqdm
from format_prompt import format_input_context

class RAGPipeline:
    def __init__(self, llm: pipeline, knowledge_index: VectorStoreFAISS = None, reranker: Optional[RAGPretrainedModel] = None):
        self.llm = llm
        self.knowledge_index = knowledge_index
        self.reranker = reranker

    def answer_batch(self, dataset_consultas, column, batch_size=100, llm_batch_size = 20, num_retrieved_docs: int = 30, num_docs_final: int = 3):
        if self.knowledge_index:
            relevant_docs = self.knowledge_index.buscar_por_batches(dataset_consultas=dataset_consultas, column=column, top_k=num_retrieved_docs, batch_size=batch_size)
            if self.reranker:
                relevant_docs =  self.rerank_documents(relevant_docs, top_k_final=num_docs_final)
        prompts = []
        valid_options_question = []
        q_id = []
        correct_answers = []
        for i in tqdm(range(len(dataset_consultas)), desc="Generando prompts"):
            if self.knowledge_index:
                context = "".join([f"\nDocument {str(i)}:" + doc for i, doc in enumerate(relevant_docs[i][1])])
                final_prompt, valid_options = format_input_context(dataset_consultas[i], context)
                prompts.append(final_prompt)
                valid_options_question.append(valid_options)
            else :
                final_prompt, valid_options = format_input_context(dataset_consultas[i])
                prompts.append(final_prompt)
                valid_options_question.append(valid_options)
            q_id.append(dataset_consultas[i]['question_id'])
            correct_answers.append(dataset_consultas[i]['answer'])
        answers = self.llm(prompts, batch_size=llm_batch_size)
        return q_id, answers, valid_options_question, correct_answers

    def rerank_documents(self, relevant_docs, top_k_final=3):
        reranked_documents = []
        for i in range(len(relevant_docs)):
          question = relevant_docs[i][0]
          docs_val = relevant_docs[i][1]
          docs = [doc[0] for doc in docs_val]  # Keep only the text
          rerank_docs = self.reranker.rerank(question, docs, k=top_k_final)
          rerank_docs = [doc["content"] for doc in rerank_docs]
          reranked_documents.append((question,rerank_docs))
        return reranked_documents

