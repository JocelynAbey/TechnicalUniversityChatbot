import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Train the College Enquiry Chatbot model'

    def handle(self, *args, **options):
        try:
            # Load dataset
            with open("chatbot_data.json", "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            # Generate embeddings
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            questions = [q["question"] for q in qa_data]
            embeddings = embedder.encode(questions, convert_to_numpy=True)

            # Build NearestNeighbors index
            index = NearestNeighbors(n_neighbors=5, metric="cosine")
            index.fit(embeddings)

            # Save model
            np.save("chatbot_embeddings.npy", embeddings)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ Model trained successfully with {len(qa_data)} Q&A pairs!"
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error training model: {e}")
            )