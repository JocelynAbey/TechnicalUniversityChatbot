import argparse
from pathlib import Path
import subprocess
from typing import Optional
import os

from .config import data_dir, data_path, dataset_path, model_path


def train_command(verbose: bool = True) -> int:
    from .core.rag import CollegeEnquiryRAGChatbot

    try:
        if verbose:
            print("🔄 Initializing chatbot...")
        chatbot = CollegeEnquiryRAGChatbot(
            data_path=dataset_path(),
            model_path=model_path(),
        )
        if verbose:
            print("📥 Loading dataset...")
        if not chatbot.load_data():
            if verbose:
                print("⚠️ Could not load dataset")
            return 1
        if verbose:
            print("🧠 Training embeddings...")
        if not chatbot.train_model():
            if verbose:
                print("⚠️ Training failed")
            return 1
        if verbose:
            print("💾 Saving model outputs...")
        if not chatbot.save_model(data_output_path=data_path()):
            if verbose:
                print("⚠️ Saving failed")
            return 1
    except Exception as exc:
        if verbose:
            print(f"⚠️ Training failed with error: {exc}")
        return 1

    if verbose:
        print("✅ Chatbot training completed and saved!")
    return 0


def chat_command(question: Optional[str]) -> int:
    from .core.rag import CollegeEnquiryRAGChatbot

    chatbot = CollegeEnquiryRAGChatbot(
        data_path=data_path(),
        model_path=model_path(),
    )
    if not chatbot.load_model():
        print("⚠️ Model not found. Please train first using the training command.")
        return 1
    if question:
        response = chatbot.chat(question)
        print(f"Bot: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Category: {response['category']}")
        if response.get("matched_question"):
            print(f"Matched Question: {response['matched_question']}")
        if response.get("related_questions"):
            print("Related Questions:", ", ".join(response["related_questions"]))
        return 0

    print("🔹 Interactive Chatbot Session 🔹")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            print("Please ask something!")
            continue
        response = chatbot.chat(user_input)
        print(f"Bot: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Category: {response['category']}")
        if response.get("matched_question"):
            print(f"Matched Question: {response['matched_question']}")
        if response.get("related_questions"):
            print("Related Questions:", ", ".join(response["related_questions"]))
        print("-" * 50)
    return 0


def django_command(project_path: Path, args: list[str]) -> int:
    manage_py = project_path / "manage.py"
    if not manage_py.exists():
        print(f"⚠️ manage.py not found at {manage_py}")
        return 1
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(
        ["python", str(manage_py), *args],
        cwd=str(project_path),
        env=env,
    )


def convert_command(
    pdf_path: Path,
    output: Optional[Path],
    output_format: str,
    method: str,
    enhance: bool,
) -> int:
    from .tools.pdf_converter import PDFToDatasetConverter

    converter = PDFToDatasetConverter()
    output_path = output
    if output_path is None:
        filename = "chatbot_dataset.json" if output_format == "json" else "chatbot_dataset.csv"
        output_path = data_dir() / filename
    try:
        dataset = converter.convert_pdf_to_dataset(
            pdf_path=str(pdf_path),
            output_path=str(output_path),
            output_format=output_format,
            extraction_method=method,
            enhance=enhance,
        )
    except Exception as exc:
        print(f"⚠️ Conversion failed: {exc}")
        return 1

    print("✅ PDF conversion completed!")
    print(f"Output file: {output_path}")
    print(f"Total Q&A pairs: {len(dataset)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="College Enquiry Chatbot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the RAG model")

    chat_parser = subparsers.add_parser("chat", help="Chat with the bot")
    chat_parser.add_argument("question", nargs="?", help="Ask a single question")

    ui_parser = subparsers.add_parser("ui", help="Manage the main Django UI")
    ui_parser.add_argument("action", choices=["migrate", "serve"], help="UI action")

    admin_parser = subparsers.add_parser("adminui", help="Manage the admin Django UI")
    admin_parser.add_argument("action", choices=["migrate", "serve"], help="Admin UI action")

    convert_parser = subparsers.add_parser("convert", help="Convert a PDF to dataset JSON/CSV")
    convert_parser.add_argument("pdf_path", type=Path, help="Path to input PDF file")
    convert_parser.add_argument("-o", "--output", type=Path, help="Output file path")
    convert_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    convert_parser.add_argument(
        "-m",
        "--method",
        choices=["auto", "pymupdf", "pdfplumber", "pypdf2"],
        default="auto",
        help="PDF extraction method (default: auto)",
    )
    convert_parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip dataset enhancement",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return train_command(verbose=True)
    if args.command == "chat":
        return chat_command(args.question)
    if args.command == "ui":
        project_path = Path(__file__).resolve().parents[2] / "web" / "collegeenquiry_chatbot"
        if args.action == "migrate":
            return django_command(project_path, ["migrate"])
        return django_command(project_path, ["runserver"])
    if args.command == "adminui":
        project_path = Path(__file__).resolve().parents[2] / "web" / "chatbot_with_admin" / "college_chatbot"
        if args.action == "migrate":
            return django_command(project_path, ["migrate"])
        return django_command(project_path, ["runserver"])
    if args.command == "convert":
        return convert_command(
            pdf_path=args.pdf_path,
            output=args.output,
            output_format=args.format,
            method=args.method,
            enhance=not args.no_enhance,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())