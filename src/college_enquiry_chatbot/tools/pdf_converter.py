import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError as exc:
    raise ImportError(
        "PDF conversion dependencies are missing. Install PyPDF2, pdfplumber, and PyMuPDF."
    ) from exc


logger = logging.getLogger(__name__)


class PDFToDatasetConverter:
    """Converts PDF documents containing Q&A pairs into chatbot training datasets."""

    def __init__(self) -> None:
        self.qa_patterns = [
            r"Q(\d+):\s*(.+?)\s*A\1:\s*(.+?)(?=Q\d+:|$)",
            r"Q\s*:\s*(.+?)\s*A\s*:\s*(.+?)(?=Q\s*:|$)",
            r"Question\s*:\s*(.+?)\s*Answer\s*:\s*(.+?)(?=Question\s*:|$)",
            r"([^.!?]*\?)\s*([^?]+?)(?=[^.!?]*\?|$)",
        ]

        self.category_keywords = {
            "departments": ["department", "dept", "faculty", "school", "division", "cse", "ece", "it", "civil", "ere"],
            "fees": ["fee", "cost", "tuition", "payment", "charge", "price", "amount", "money", "rupees", "₹"],
            "facilities": ["library", "canteen", "hostel", "parking", "gym", "lab", "toilet", "restroom", "wifi"],
            "contact": ["contact", "phone", "email", "address", "office", "reception", "call", "reach"],
            "admissions": ["admission", "application", "entrance", "exam", "keam", "apply", "eligibility"],
            "programs": ["program", "course", "degree", "btech", "mtech", "diploma", "engineering"],
            "location": ["where", "located", "address", "place", "building", "floor", "room"],
            "timing": ["time", "hours", "schedule", "timing", "open", "close", "when"],
            "leadership": ["principal", "hod", "head", "director", "dean", "faculty", "professor"],
            "placements": ["placement", "job", "career", "company", "recruiter", "employment"],
            "activities": ["club", "association", "society", "event", "activity", "sports"],
        }

    def extract_text_pypdf2(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            logger.error("PyPDF2 extraction failed: %s", exc)
            return ""

    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as exc:
            logger.error("pdfplumber extraction failed: %s", exc)
            return ""

    def extract_text_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            try:
                return "\n".join(page.get_text() for page in doc)
            finally:
                doc.close()
        except Exception as exc:
            logger.error("PyMuPDF extraction failed: %s", exc)
            return ""

    def extract_text_from_pdf(self, pdf_path: str, method: str = "auto") -> str:
        logger.info("Extracting text from %s using method: %s", pdf_path, method)

        if method == "auto":
            methods = [
                ("pymupdf", self.extract_text_pymupdf),
                ("pdfplumber", self.extract_text_pdfplumber),
                ("pypdf2", self.extract_text_pypdf2),
            ]
            for method_name, extract_func in methods:
                text = extract_func(pdf_path)
                if text.strip():
                    logger.info("Successfully extracted text using %s", method_name)
                    return text
            logger.error("All extraction methods failed")
            return ""

        if method == "pymupdf":
            return self.extract_text_pymupdf(pdf_path)
        if method == "pdfplumber":
            return self.extract_text_pdfplumber(pdf_path)
        if method == "pypdf2":
            return self.extract_text_pypdf2(pdf_path)
        raise ValueError(f"Unknown extraction method: {method}")

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\d+\n", "\n", text)
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        qa_pairs: List[Tuple[str, str]] = []
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) == 3:
                    question, answer = match[1], match[2]
                elif len(match) == 2:
                    question, answer = match
                else:
                    continue
                question = self.clean_qa_text(question)
                answer = self.clean_qa_text(answer)
                if self.is_valid_qa_pair(question, answer):
                    qa_pairs.append((question, answer))

        seen = set()
        unique_pairs = []
        for question, answer in qa_pairs:
            qa_key = (question.lower().strip(), answer.lower().strip())
            if qa_key not in seen:
                seen.add(qa_key)
                unique_pairs.append((question, answer))

        logger.info("Extracted %s unique Q&A pairs", len(unique_pairs))
        return unique_pairs

    def clean_qa_text(self, text: str) -> str:
        text = text.strip()
        prefixes = ["Q:", "A:", "Question:", "Answer:", "Ans:", "•", "-", "*"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        text = text.rstrip(":").strip()
        if "?" in text and not text.endswith("?"):
            text = text.replace("?", "") + "?"
        return text

    def is_valid_qa_pair(self, question: str, answer: str) -> bool:
        if len(question) < 5 or len(answer) < 5:
            return False
        if len(question) > 500 or len(answer) > 2000:
            return False
        question_indicators = [
            "what",
            "where",
            "when",
            "how",
            "who",
            "why",
            "which",
            "can",
            "is",
            "are",
            "do",
            "does",
            "?",
        ]
        if not any(indicator in question.lower() for indicator in question_indicators):
            return False
        if len(answer.split()) < 3:
            return False
        return True

    def categorize_qa_pair(self, question: str, answer: str) -> str:
        text = (question + " " + answer).lower()
        category_scores: Dict[str, int] = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        return max(category_scores, key=category_scores.get) if category_scores else "general"

    def create_dataset(self, qa_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        dataset: List[Dict[str, str]] = []
        for question, answer in qa_pairs:
            dataset.append(
                {
                    "question": question,
                    "answer": answer,
                    "category": self.categorize_qa_pair(question, answer),
                }
            )
        return dataset

    def enhance_dataset(self, dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
        enhanced_dataset = list(dataset)
        variations: List[Dict[str, str]] = []
        for item in dataset:
            question = item["question"]
            for variation in self.generate_question_variations(question):
                if variation != question:
                    variations.append(
                        {
                            "question": variation,
                            "answer": item["answer"],
                            "category": item["category"],
                        }
                    )
        enhanced_dataset.extend(variations)
        logger.info("Enhanced dataset from %s to %s entries", len(dataset), len(enhanced_dataset))
        return enhanced_dataset

    def generate_question_variations(self, question: str) -> List[str]:
        variations = [question]
        transformations = [
            (r"^What is ", ""),
            (r"^Where is ", ""),
            (r"^How can I ", ""),
            (r"\?$", ""),
            ("", "Tell me about "),
            ("", "Information about "),
            ("", "Details about "),
        ]
        for old_pattern, new_pattern in transformations:
            if old_pattern:
                new_question = re.sub(old_pattern, new_pattern, question, flags=re.IGNORECASE)
            else:
                new_question = new_pattern + question
            if new_question != question:
                variations.append(new_question)
        return variations

    def save_dataset(self, dataset: List[Dict[str, str]], output_path: str, output_format: str = "json") -> None:
        if output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        elif output_format == "csv":
            df = pd.DataFrame(dataset)
            df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        logger.info("Dataset saved to %s (%s entries)", output_path, len(dataset))

    def generate_statistics(self, dataset: List[Dict[str, str]]) -> Dict[str, float]:
        if not dataset:
            return {}
        category_counts: Dict[str, int] = {}
        for category in (item["category"] for item in dataset):
            category_counts[category] = category_counts.get(category, 0) + 1
        avg_question_length = sum(len(item["question"]) for item in dataset) / len(dataset)
        avg_answer_length = sum(len(item["answer"]) for item in dataset) / len(dataset)
        return {
            "total_pairs": len(dataset),
            "categories": category_counts,
            "avg_question_length": round(avg_question_length, 2),
            "avg_answer_length": round(avg_answer_length, 2),
        }

    def convert_pdf_to_dataset(
        self,
        pdf_path: str,
        output_path: str | None = None,
        output_format: str = "json",
        extraction_method: str = "auto",
        enhance: bool = True,
    ) -> List[Dict[str, str]]:
        logger.info("Starting PDF to dataset conversion")
        text = self.extract_text_from_pdf(pdf_path, extraction_method)
        if not text:
            raise ValueError("Failed to extract text from PDF")
        cleaned_text = self.clean_text(text)
        qa_pairs = self.extract_qa_pairs(cleaned_text)
        if not qa_pairs:
            raise ValueError("No Q&A pairs found in PDF")
        dataset = self.create_dataset(qa_pairs)
        if enhance:
            dataset = self.enhance_dataset(dataset)
        if output_path:
            self.save_dataset(dataset, output_path, output_format)
        stats = self.generate_statistics(dataset)
        logger.info("Dataset statistics: %s", stats)
        return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDF to chatbot dataset")
    parser.add_argument("pdf_path", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["auto", "pymupdf", "pdfplumber", "pypdf2"],
        default="auto",
        help="PDF extraction method (default: auto)",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Skip dataset enhancement")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    output = args.output
    if not output:
        output = "chatbot_dataset.json" if args.format == "json" else "chatbot_dataset.csv"
    converter = PDFToDatasetConverter()
    dataset = converter.convert_pdf_to_dataset(
        pdf_path=args.pdf_path,
        output_path=output,
        output_format=args.format,
        extraction_method=args.method,
        enhance=not args.no_enhance,
    )
    print("Successfully converted PDF to dataset!")
    print(f"Output file: {output}")
    print(f"Total Q&A pairs: {len(dataset)}")
    print("\nSample entries:")
    for i, item in enumerate(dataset[:3]):
        print(f"{i + 1}. Category: {item['category']}")
        print(f"   Q: {item['question'][:80]}...")
        print(f"   A: {item['answer'][:80]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())