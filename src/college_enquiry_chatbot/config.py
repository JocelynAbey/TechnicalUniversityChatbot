from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def dataset_path() -> Path:
    return data_dir() / "chatbot_dataset.json"


def data_path() -> Path:
    return data_dir() / "chatbot_data.json"


def model_path() -> Path:
    return data_dir() / "chatbot_embeddings.npy"