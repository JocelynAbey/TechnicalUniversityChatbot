# College Enquiry Chatbot

This project provides a **RAG + LLM chatbot** for college enquiries, exposed as both a **CLI** and a **Django web app**.

## Features
- Sentence-transformer embeddings + NearestNeighbors retrieval
- Flan-T5 answer generation
- CLI for training and chatting
- Django UI + API with chat history

## Install (editable)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## CLI Usage
Train the model:
```bash
college-chatbot train
```

Chat interactively:
```bash
college-chatbot chat
```

Ask a single question:
```bash
college-chatbot chat "What courses are offered?"
```

## Django Web App
```bash
cd web/collegeenquiry_chatbot
python manage.py migrate
python manage.py runserver
```

Open: http://127.0.0.1:8000/

## Data Files
Shared assets live in `data/`:
- `chatbot_dataset.json`
- `chatbot_data.json`
- `chatbot_embeddings.npy`

If you update `chatbot_dataset.json`, re-run `college-chatbot train`.