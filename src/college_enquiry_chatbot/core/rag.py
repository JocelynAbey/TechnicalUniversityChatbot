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
        self.llm = self._build_llm_pipeline(llm_name)

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
        if self._is_greeting(user_question):
            return {
                "answer": "Hello! How can I help you with college enquiries today?",
                "confidence": 1.0,
                "category": "greeting",
                "matched_question": None,
                "related_questions": [],
            }
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
        ].strip()

        answer_text = self._strip_prompt(prompt, llm_response)
        if sim >= 0.9:
            answer_text = best_item.answer
        elif not self._uses_context(answer_text, best_item.answer):
            answer_text = best_item.answer

        return {
            "answer": answer_text,
            "confidence": round(sim, 2),
            "category": best_item.category,
            "matched_question": best_item.question,
            "related_questions": [
                item.question for item in context_items if item.question != best_item.question
            ],
        }

    def chat(self, user_question: str) -> dict:
        return self.generate_answer(user_question)

    @staticmethod
    def _is_greeting(text: str) -> bool:
        normalized = text.strip().lower()
        greetings = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "namaste",
            "hola",
        }
        return normalized in greetings

    @staticmethod
    def _uses_context(answer: str, context_answer: str) -> bool:
        answer_lower = answer.lower()
        context_lower = context_answer.lower()
        if not answer_lower:
            return False
        shared_tokens = {
            token
            for token in context_lower.split()
            if token.isalpha() and token in answer_lower
        }
        return len(shared_tokens) >= 2

    @staticmethod
    def _strip_prompt(prompt: str, generated: str) -> str:
        if "Answer:" in generated:
            return generated.split("Answer:")[-1].strip()
        if generated.startswith(prompt):
            return generated[len(prompt):].strip()
        return generated

    @staticmethod
    def _build_llm_pipeline(model_name: str):
        try:
            return pipeline("text2text-generation", model=model_name)
        except KeyError:
            return pipeline("text-generation", model=model_name)