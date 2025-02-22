import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging

def create_evaluator_information_retrieval(test_dataset):
    logging.info("Creating evaluator...")
    corpus = dict(zip(test_dataset["q_id"], test_dataset["positive"]))
    queries = dict(zip(test_dataset["q_id"], test_dataset["anchor"]))

    relevant_docs = {}
    for q_id in queries:
        relevant_docs[q_id] = [q_id]

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        batch_size=16,
        name="telecom-ir-eval",
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1],
        ndcg_at_k=[10],
        mrr_at_k=[10],
    )
    logging.info("Evaluator created.")
    return evaluator


