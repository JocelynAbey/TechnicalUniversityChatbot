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

## Step-by-step: Run the CLI
1. Ensure data files exist in `data/` (already provided).
2. Train the model:
   ```bash
   college-chatbot train
   ```
3. Start a chat session:
   ```bash
   college-chatbot chat
   ```
   Or ask a single question:
   ```bash
   college-chatbot chat "What courses are offered?"
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

## Step-by-step: Run the Django App
1. Activate your virtual environment and install deps.
2. From the project root, run:
   ```bash
   cd web/collegeenquiry_chatbot
   python manage.py migrate
   python manage.py runserver
   ```
3. Open: http://127.0.0.1:8000/

Or use the CLI shortcuts:
```bash
college-chatbot ui migrate
college-chatbot ui serve
```

## Admin Django App (with dataset upload/training)
```bash
cd web/chatbot_with_admin/college_chatbot
python manage.py migrate
python manage.py runserver
```

Notes:
- The admin app is configured to use SQLite by default (see `web/chatbot_with_admin/college_chatbot/college_chatbot/settings.py`).
- The first run auto-creates tables using `chatbot_app/sqlite_init.sql`.
- Default admin credentials: **admin / admin**

Or use the CLI shortcuts:
```bash
college-chatbot adminui migrate
college-chatbot adminui serve
```

## Data Files
Shared assets live in `data/`:
- `chatbot_dataset.json`
- `chatbot_data.json`
- `chatbot_embeddings.npy`

If you update `chatbot_dataset.json`, re-run `college-chatbot train`.