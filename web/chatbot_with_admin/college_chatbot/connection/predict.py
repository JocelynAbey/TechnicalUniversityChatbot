import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline


class LBSITWRAGChatbot:
    def __init__(self, data_path="chatbot_dataset.json"):
        self.data_path = data_path
        self.qa_data = []
        self.model_trained = False

        # RAG Component: Initialize embedding model for semantic similarity (used for retrieval in RAG)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # LLM Component: Initialize text generation model (used for answer generation in RAG)
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base")

        # RAG Component: Retrieval index for storing embeddings
        self.index = None
        self.embeddings = None

    def load_model(self, model_path="chatbot_embeddings.npy", data_path="chatbot_data.json"):
        """Load RAG embeddings and dataset"""
        try:
            if os.path.exists(model_path) and os.path.exists(data_path):
                self.embeddings = np.load(model_path)
                with open(data_path, "r", encoding="utf-8") as f:
                    self.qa_data = json.load(f)

                # RAG Component: Rebuild NearestNeighbors index for retrieval
                self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
                self.index.fit(self.embeddings)

                self.model_trained = True
                print("✅ Model loaded successfully!")
                return True
            else:
                print("⚠️ Saved model files not found.")
                return False
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return False

    def retrieve_context(self, user_question, top_k=3):
        """RAG Component: Retrieve top-k similar Q&A pairs for context"""
        if not self.model_trained:
            return []

        # RAG Component: Encode user question into embedding
        q_emb = self.embedder.encode([user_question], convert_to_numpy=True)
        
        # RAG Component: Find nearest neighbors (retrieve relevant Q&A pairs)
        distances, indices = self.index.kneighbors(q_emb, n_neighbors=top_k)

        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.qa_data):
                retrieved.append({
                    "question": self.qa_data[idx]["question"],
                    "answer": self.qa_data[idx]["answer"],
                    "category": self.qa_data[idx].get("category", "general"),
                    "similarity": float(1 - dist)  # cosine similarity
                })
        return retrieved

    def generate_answer(self, user_question, threshold=0.6):
        """RAG + LLM Component: Combines retrieval (RAG) and answer generation (LLM)"""
        # RAG Component: Fetch relevant context using retrieval
        context_items = self.retrieve_context(user_question, top_k=3)
        if not context_items:
            return {
                "answer": "I'm sorry, I couldn't find relevant information.",
                "confidence": 0.0,
                "category": "unknown",
                "matched_question": None
            }

        # RAG Component: Select most relevant item and apply similarity threshold
        best_item = max(context_items, key=lambda x: x["similarity"])
        sim = best_item["similarity"]

        if sim < threshold:
            return {
                "answer": "I'm sorry, I couldn't find relevant information.",
                "confidence": round(sim, 2),
                "category": "unknown",
                "matched_question": None,
                "related_questions": []
            }

        # RAG Component: Build context text from retrieved items for LLM
        context_text = "\n".join([f"Q: {item['question']} A: {item['answer']}" for item in context_items])

        # RAG + LLM Component: Construct prompt combining retrieved context (RAG) and user question for LLM
        prompt = f"""
You are a helpful assistant for LBSITW college.
Use the following context to answer the question truthfully.

Context:
{context_text}

Question: {user_question}
Answer:
"""

        # LLM Component: Generate answer using the LLM with the provided context
        llm_response = self.llm(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]

        # RAG + LLM Component: Return response combining LLM-generated answer and RAG metadata
        return {
            "answer": llm_response.strip(),
            "confidence": round(sim, 2),
            "category": best_item.get("category", "general"),
            "matched_question": best_item["question"],
            "related_questions": [item["question"] for item in context_items if item["question"] != best_item["question"]]
        }

    def chat(self, user_question):
        """RAG + LLM Component: Entry point for chat, combining RAG retrieval and LLM generation"""
        return self.generate_answer(user_question)


def interactive_chat():
    """RAG + LLM Component: Interactive console for chatbot predictions using RAG and LLM"""
    chatbot = LBSITWRAGChatbot()
    if chatbot.load_model("chatbot_embeddings.npy", "chatbot_data.json"):
        print("🔹 Interactive Chatbot Session 🔹")
        print("Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            if not user_input:
                print("Please ask something!")
                continue
            # RAG + LLM Component: Generate response using RAG retrieval and LLM generation
            response = chatbot.chat(user_input)
            print(f"Bot: {response['answer']}")
            print(f"Confidence: {response['confidence']}")
            print(f"Category: {response['category']}")
            if response['matched_question']:
                print(f"Matched Question: {response['matched_question']}")
            if response['related_questions']:
                print("Related Questions:", ", ".join(response['related_questions']))
            print("-" * 50)
    else:
        print("⚠️ Model not found. Please train first using the training script.")


if __name__ == "__main__":
    interactive_chat()