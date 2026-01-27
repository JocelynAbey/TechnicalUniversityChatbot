from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline


@dataclass
class RetrievalItem:
    question: str
    answer: str
    category: str
    similarity: float


class CollegeEnquiryRAGChatbot:
    def __init__(
        self,
        data_path: Path,
        model_path: Path,
        embedder_name: str = "all-MiniLM-L6-v2",
        llm_name: str = "google/flan-t5-base",
    ) -> None:
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.qa_data = []
        self.model_trained = False

        self.embedder = SentenceTransformer(embedder_name)
        self.llm = pipeline("text2text-generation", model=llm_name)

        self.index = None
        self.embeddings = None

    def load_model(self) -> bool:
        if not self.model_path.exists() or not self.data_path.exists():
            return False
        try:
            self.embeddings = np.load(self.model_path)
            with self.data_path.open("r", encoding="utf-8") as f:
                self.qa_data = json.load(f)
            self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
            self.index.fit(self.embeddings)
            self.model_trained = True
            return True
        except Exception:
            return False

    def load_data(self) -> bool:
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                self.qa_data = json.load(f)
            return True
        except Exception:
            return False

    def train_model(self) -> bool:
        if not self.qa_data:
            return False
        questions = [q["question"] for q in self.qa_data]
        self.embeddings = self.embedder.encode(questions, convert_to_numpy=True)
        self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.index.fit(self.embeddings)
        self.model_trained = True
        return True

    def save_model(self) -> bool:
        if not self.model_trained or self.embeddings is None:
            return False
        try:
            np.save(self.model_path, self.embeddings)
            with self.data_path.open("w", encoding="utf-8") as f:
                json.dump(self.qa_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def retrieve_context(self, user_question: str, top_k: int = 3) -> List[RetrievalItem]:
        if not self.model_trained or self.index is None:
            return []
        q_emb = self.embedder.encode([user_question], convert_to_numpy=True)
        distances, indices = self.index.kneighbors(q_emb, n_neighbors=top_k)

        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.qa_data):
                retrieved.append(
                    RetrievalItem(
                        question=self.qa_data[idx]["question"],
                        answer=self.qa_data[idx]["answer"],
                        category=self.qa_data[idx].get("category", "general"),
                        similarity=float(1 - dist),
                    )
                )
        return retrieved

    def generate_answer(self, user_question: str, threshold: float = 0.6) -> dict:
        context_items = self.retrieve_context(user_question, top_k=3)
        if not context_items:
            return {
                "answer": "I'm sorry, I couldn't find relevant information in our college database.",
                "confidence": 0.0,
                "category": "unknown",
                "matched_question": None,
                "related_questions": [],
            }

        best_item = max(context_items, key=lambda x: x.similarity)
        sim = best_item.similarity
        if sim < threshold:
            return {
                "answer": "I'm sorry, I couldn't find relevant information in our college database.",
                "confidence": round(sim, 2),
                "category": "unknown",
                "matched_question": None,
                "related_questions": [],
            }

        context_text = "\n".join(
            [f"Q: {item.question} A: {item.answer}" for item in context_items]
        )
        prompt = f"""
You are a helpful assistant for college enquiries.
Use the following context to answer the question truthfully.

Context:
{context_text}

Question: {user_question}
Answer:
"""

        llm_response = self.llm(prompt, max_length=150, num_return_sequences=1)[0][
            "generated_text"
        ]

        return {
            "answer": llm_response.strip(),
            "confidence": round(sim, 2),
            "category": best_item.category,
            "matched_question": best_item.question,
            "related_questions": [
                item.question for item in context_items if item.question != best_item.question
            ],
        }

    def chat(self, user_question: str) -> dict:
        return self.generate_answer(user_question)