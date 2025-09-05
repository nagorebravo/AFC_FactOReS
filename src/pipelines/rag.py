import logging
from typing import Dict, List, Optional

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.api_calls.api_manager import ApiManager
from src.pipelines.utils import cosine_similarity


class RAG:
    def __init__(self, api_manager: Optional[ApiManager] = None):
        self.api_manager = api_manager or ApiManager()
        self.metadatas = []
        self.documents = []
        self.questions = []

    def __len__(self):
        return len(self.documents)

    def add_documents(self, documents: List[Dict[str, str]]):
        self.metadatas.extend(
            [
                {"url": x["url"], "favicon": x["favicon"], "source": x["source"]}
                for x in documents
            ]
        )
        self.documents.extend(x["text"] for x in documents)

    def search_questions(
        self,
        questions: List[str],
        top_k: int = 5,
        search_engine: str = "serpapi",
        language: str = "es",
        location: str = None,
    ):
        docs = self.api_manager.web_search(
            model=search_engine,
            queries=questions,
            top_k=top_k,
            language=language,
            location=location,
            ban_domains=None,
        )

        self.add_documents(docs)
        self.questions.extend(questions)

    @torch.no_grad()
    def generate(
        self,
        embedding_model: str = "openai_embeddings_large",
        top_k: int = 3,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        questions: Optional[List[str]] = None,
        batch_size: int = 256,
        split_separators: List[str] = ["\n"],
    ):
        """
        IMPORTANT: This question will re-compute the embeddings for the questions and documents each time it is called.
        It is designed for easily development and testing of hyperparameters. In the future, if we need to call the
        function multiple times with different questions, we should refactor the code to compute the document embeddings
        in add_documents and only compute the question embeddings in this function.
        """

        if not self.documents:
            #raise ValueError("No documents added to the pipeline.")
            logging.warning("No documents added to the pipeline.")
            return
        if not self.questions and questions is None:
            raise ValueError("No questions added to the pipeline.")

        gen_questions = questions or self.questions

        text_splitter = RecursiveCharacterTextSplitter(
            separators=split_separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
        )

        splits = text_splitter.create_documents(
            texts=self.documents, metadatas=self.metadatas
        )

        splits = [x for x in splits if len(x.page_content) < 4000]
        docs, metadata = zip(*((x.page_content, x.metadata) for x in splits))
        if "rerank" not in embedding_model:
            questions_embeddings, documents_embeddings = self.api_manager.embeddings(
                model=embedding_model,
                queries=gen_questions,
                passages=docs,
                batch_size=batch_size,
            )

            similarities = cosine_similarity(questions_embeddings, documents_embeddings)
        else:
            similarities = self.api_manager.rerank(
                model=embedding_model,
                queries=gen_questions,
                passages=docs,
            )
        top_k_indices = torch.topk(similarities, k=top_k, dim=1).indices

        results = {
            question: [
                {"metadata": metadata[idx], "text": docs[idx]} for idx in indices
            ]
            for question, indices in zip(gen_questions, top_k_indices)
        }

        # Calculate and log the percentage of documents used
        total_snippets = len(docs)
        snippets_used = len(
            set(idx.item() for indices in top_k_indices for idx in indices)
        )
        snippets_percentage = (snippets_used / total_snippets) * 100

        total_documents = len(self.documents)
        documents_used = len(
            set(
                metadata[idx.item()]["url"]
                for indices in top_k_indices
                for idx in indices
            )
        )
        documents_percentage = (documents_used / total_documents) * 100

        logging.info(
            f"Percentage of snippets used: {snippets_percentage:.2f}% ({snippets_used}/{total_snippets})"
        )
        logging.info(
            f"Percentage of original documents used: {documents_percentage:.2f}% ({documents_used}/{total_documents})"
        )

        return results

    @torch.no_grad()
    def generate_w_ranks(
        self,
        embedding_model: str = "openai_embeddings_large",
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        questions: Optional[List[str]] = None,
        batch_size: int = 256,
        split_separators: List[str] = ["\n"],
    ):
        """
        IMPORTANT: This question will re-compute the embeddings for the questions and documents each time it is called.
        It is designed for easily development and testing of hyperparameters. In the future, if we need to call the
        function multiple times with different questions, we should refactor the code to compute the document embeddings
        in add_documents and only compute the question embeddings in this function.
        """

        if not self.documents:
            #raise ValueError("No documents added to the pipeline.")
            logging.warning("No documents added to the pipeline.")
            return
        if not self.questions and questions is None:
            raise ValueError("No questions added to the pipeline.")

        gen_questions = questions or self.questions

        text_splitter = RecursiveCharacterTextSplitter(
            separators=split_separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
        )

        splits = text_splitter.create_documents(
            texts=self.documents, metadatas=self.metadatas
        )

        splits = [x for x in splits if len(x.page_content) < 4000]
        docs, metadata = zip(*((x.page_content, x.metadata) for x in splits))
        if "rerank" not in embedding_model:
            questions_embeddings, documents_embeddings = self.api_manager.embeddings(
                model=embedding_model,
                queries=gen_questions,
                passages=docs,
                batch_size=batch_size,
            )

            similarities = cosine_similarity(questions_embeddings, documents_embeddings)
        else:
            similarities = self.api_manager.rerank(
                model=embedding_model,
                queries=gen_questions,
                passages=docs,
            )
        sorted_similarities, sorted_indices = torch.sort(similarities, dim=1, descending=True)

        results = {
            question: [
                {
                    "metadata": metadata[idx.item()], 
                    "text": docs[idx.item()],
                    "score": score.item()
                } 
                for idx, score in zip(indices, sims)
            ]
            for question, indices, sims in zip(gen_questions, sorted_indices, sorted_similarities)
        }

        return results

    def __call__(
        self,
        questions: List[str],
        top_k: int = 3,
        embedding_model: str = "openai_embeddings_large",
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        batch_size: int = 256,
        split_separators: List[str] = ["\n"],
    ):
        self.search_questions(questions)
        return self.generate(
            questions=questions,
            top_k=top_k,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            split_separators=split_separators,
        )


if __name__ == "__main__":
    import time

    rag = RAG()
    questions = ["What is the capital of Spain?"]
    start = time.time()
    rag.search_questions(questions)
    results = rag.generate(
        embedding_model="openai_embeddings_small",
        top_k=2,
        chunk_size=32,
        chunk_overlap=8,
    )
    total_time = time.time() - start

    results = results[questions[0]]
    docs = [x["text"] for x in results]
    metadatas = [x["metadata"] for x in results]
    print(f"RAG results. What is the capital of Spain?: {docs}")

    print(f"Total time in seconds: {round(total_time, 2)}")
