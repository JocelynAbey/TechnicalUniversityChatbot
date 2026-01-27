import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class LBSITWRAGChatbot:
    def __init__(self, data_path="chatbot_dataset.json"):
        self.data_path = data_path
        self.qa_data = []
        self.model_trained = False

        # RAG Component: Initialize embedding model for semantic similarity (used for retrieval in RAG)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # RAG Component: Retrieval index for storing embeddings
        self.index = None
        self.embeddings = None

    def load_data(self):
        """Load dataset from JSON file for RAG retrieval"""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.qa_data = json.load(f)
            print(f"✅ Loaded {len(self.qa_data)} Q&A pairs")
            return True
        except Exception as e:
            print(f"⚠️ Error loading dataset: {e}")
            return False

    def train_model(self):
        """RAG Component: Build NearestNeighbors index for retrieval"""
        if not self.qa_data:
            print("⚠️ No data loaded.")
            return False

        # RAG Component: Generate embeddings for questions using SentenceTransformer
        questions = [q["question"] for q in self.qa_data]
        self.embeddings = self.embedder.encode(questions, convert_to_numpy=True)

        # RAG Component: Fit NearestNeighbors index for fast retrieval
        self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.index.fit(self.embeddings)

        self.model_trained = True
        print("✅ RAG model trained with NearestNeighbors")
        return True

    def save_model(self, model_path="chatbot_embeddings.npy", data_path="chatbot_data.json"):
        """Save RAG embeddings and dataset"""
        if not self.model_trained or self.index is None:
            print("⚠️ Model not trained yet.")
            return False
        try:
            # Save embeddings and dataset
            np.save(model_path, self.embeddings)
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Model saved: {model_path}, {data_path}")
            return True
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
            return False


def train_chatbot():
    """RAG Component: Train the RAG model"""
    chatbot = LBSITWRAGChatbot()
    if not chatbot.load_data():
        print("⚠️ Could not load dataset")
        return False
    if chatbot.train_model():
        chatbot.save_model("chatbot_embeddings.npy", "chatbot_data.json")
        print("✅ Chatbot training completed and saved!")
        return True
    else:
        print("⚠️ Training failed")
        return False


if __name__ == "__main__":
    train_chatbot()