import unittest


class TestRag(unittest.TestCase):
    def test_rag(self):
        from src.pipelines.rag import RAG

        rag = RAG()
        questions = ["What is the capital of Spain?"]
        rag.search_questions(questions)
        results = rag.generate(
            embedding_model="openai_embeddings_small",
            top_k=2,
            chunk_size=32,
            chunk_overlap=8,
        )
        results = results[questions[0]]
        docs = [x["text"] for x in results]
        metadatas = [x["metadata"] for x in results]
        self.assertTrue(docs)
        self.assertTrue(metadatas)

        print(f"RAG results. What is the capital of Spain?: {docs}")

        print(
            f"Tested RAG pipeline with OpenAI embeddings. Cost: {rag.api_manager.cost}$"
        )
