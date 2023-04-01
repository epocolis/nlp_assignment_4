import os
import logging
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers


class Question_and_Answer_System:
    def __init__(self):
        self.pipe_line = None

    def setup_logging(self):
        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
        )
        logging.getLogger("haystack").setLevel(logging.INFO)

    def prepare_documents(self, doc_dir):
        document_store = InMemoryDocumentStore(use_bm25=True)
        doc_dir = "data"
        files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
        indexing_pipeline = TextIndexingPipeline(document_store)
        indexing_pipeline.run_batch(file_paths=files_to_index)
        return document_store

    def create_retriever(self, document_store):
        retriever = BM25Retriever(document_store=document_store)
        return retriever

    def create_reader(self, document_store):
        reader = FARMReader(
            model_name_or_path="deepset/roberta-base-squad2", use_gpu=False
        )
        return reader

    def create_pipeline(self, reader, retriever):
        self.pipe_line = ExtractiveQAPipeline(reader, retriever)
        return self.pipe_line

    def answer_question(self, question: str):
        prediction = self.pipe_line.run(
            query=question,
            params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
        )

        return prediction

    def format_answers(self, prediction):
        print_answers(
            prediction, details="minimum"  ## Choose from `minimum`, `medium`, and `all`
        )


if __name__ == "__main__":
    qa = Question_and_Answer_System()
    doc_dir = "data"
    document_store = qa.prepare_documents(doc_dir)
    reader = qa.create_reader(document_store)
    retriever = qa.create_retriever(document_store)
    pipe_line = qa.create_pipeline(reader, retriever)

    while True:
        question = input("Ask me something or enter -1 to quit:")
        if question != "-1":
            prediction = qa.answer_question(question)
            qa.format_answers(prediction)
