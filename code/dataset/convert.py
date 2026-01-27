import re
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import logging

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Installing required packages...")
    import subprocess
    import sys
    
    packages = ['PyPDF2', 'pdfplumber', 'PyMuPDF']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    import PyPDF2
    import pdfplumber
    import fitz
    PYMUPDF_AVAILABLE = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFToDatasetConverter:
    """
    Converts PDF documents containing Q&A pairs into chatbot training datasets
    """
    
    def __init__(self):
        self.qa_patterns = [
            # Pattern 1: Qnum: ... Anum: ...
            r'Q(\d+):\s*(.+?)\s*A\1:\s*(.+?)(?=Q\d+:|$)',
            # Pattern 2: Q: ... A: ...
            r'Q\s*:\s*(.+?)\s*A\s*:\s*(.+?)(?=Q\s*:|$)',
            # Pattern 3: Question: ... Answer: ...
            r'Question\s*:\s*(.+?)\s*Answer\s*:\s*(.+?)(?=Question\s*:|$)',
            # Pattern 4: Simple Q&A without prefixes
            r'([^.!?]*\?)\s*([^?]+?)(?=[^.!?]*\?|$)',
        ]
        
        # Category keywords for automatic categorization
        self.category_keywords = {
            'departments': ['department', 'dept', 'faculty', 'school', 'division', 'cse', 'ece', 'it', 'civil', 'ere'],
            'fees': ['fee', 'cost', 'tuition', 'payment', 'charge', 'price', 'amount', 'money', 'rupees', '₹'],
            'facilities': ['library', 'canteen', 'hostel', 'parking', 'gym', 'lab', 'toilet', 'restroom', 'wifi'],
            'contact': ['contact', 'phone', 'email', 'address', 'office', 'reception', 'call', 'reach'],
            'admissions': ['admission', 'application', 'entrance', 'exam', 'keam', 'apply', 'eligibility'],
            'programs': ['program', 'course', 'degree', 'btech', 'mtech', 'diploma', 'engineering'],
            'location': ['where', 'located', 'address', 'place', 'building', 'floor', 'room'],
            'timing': ['time', 'hours', 'schedule', 'timing', 'open', 'close', 'when'],
            'leadership': ['principal', 'hod', 'head', 'director', 'dean', 'faculty', 'professor'],
            'placements': ['placement', 'job', 'career', 'company', 'recruiter', 'employment'],
            'activities': ['club', 'association', 'society', 'event', 'activity', 'sports'],
        }
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fastest and most accurate)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = 'auto') -> str:
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_path: Path to PDF file
            method: 'auto', 'pymupdf', 'pdfplumber', or 'pypdf2'
        
        Returns:
            Extracted text
        """
        logger.info(f"Extracting text from {pdf_path} using method: {method}")
        
        if method == 'auto':
            # Try methods in order of preference
            methods = [
                ('pymupdf', self.extract_text_pymupdf),
                ('pdfplumber', self.extract_text_pdfplumber),
                ('pypdf2', self.extract_text_pypdf2)
            ]
            
            for method_name, extract_func in methods:
                text = extract_func(pdf_path)
                if text.strip():
                    logger.info(f"Successfully extracted text using {method_name}")
                    return text
            
            logger.error("All extraction methods failed")
            return ""
        
        elif method == 'pymupdf':
            return self.extract_text_pymupdf(pdf_path)
        elif method == 'pdfplumber':
            return self.extract_text_pdfplumber(pdf_path)
        elif method == 'pypdf2':
            return self.extract_text_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common OCR errors
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract Q&A pairs from text using multiple patterns
        
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) == 3:  # for numbered pattern
                    question, answer = match[1], match[2]
                elif len(match) == 2:
                    question, answer = match
                else:
                    continue
                
                question = self.clean_qa_text(question)
                answer = self.clean_qa_text(answer)
                
                if self.is_valid_qa_pair(question, answer):
                    qa_pairs.append((question, answer))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for q, a in qa_pairs:
            qa_key = (q.lower().strip(), a.lower().strip())
            if qa_key not in seen:
                seen.add(qa_key)
                unique_pairs.append((q, a))
        
        logger.info(f"Extracted {len(unique_pairs)} unique Q&A pairs")
        return unique_pairs
    
    def clean_qa_text(self, text: str) -> str:
        """Clean individual question or answer text"""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['Q:', 'A:', 'Question:', 'Answer:', 'Ans:', '•', '-', '*']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove trailing colons
        text = text.rstrip(':').strip()
        
        # Ensure questions end with question mark
        if '?' in text and not text.endswith('?'):
            # Move question mark to the end if it's in the middle
            text = text.replace('?', '') + '?'
        
        return text
    
    def is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """Validate if a Q&A pair is meaningful"""
        # Check minimum length
        if len(question) < 5 or len(answer) < 5:
            return False
        
        # Check maximum length (avoid extracting entire paragraphs)
        if len(question) > 500 or len(answer) > 2000:
            return False
        
        # Questions should contain question words or end with ?
        question_indicators = ['what', 'where', 'when', 'how', 'who', 'why', 'which', 'can', 'is', 'are', 'do', 'does', '?']
        if not any(indicator in question.lower() for indicator in question_indicators):
            return False
        
        # Avoid very short answers
        if len(answer.split()) < 3:
            return False
        
        return True
    
    def categorize_qa_pair(self, question: str, answer: str) -> str:
        """Automatically categorize Q&A pair based on keywords"""
        text = (question + " " + answer).lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'
    
    def create_dataset(self, qa_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        Convert Q&A pairs to chatbot dataset format
        
        Returns:
            List of dictionaries with question, answer, and category
        """
        dataset = []
        
        for question, answer in qa_pairs:
            category = self.categorize_qa_pair(question, answer)
            
            dataset.append({
                'question': question,
                'answer': answer,
                'category': category
            })
        
        return dataset
    
    def enhance_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Enhance dataset with variations and synonyms
        """
        enhanced_dataset = list(dataset)  # Copy original dataset
        
        # Add common variations
        variations = []
        for item in dataset:
            question = item['question']
            answer = item['answer']
            category = item['category']
            
            # Create question variations
            question_variations = self.generate_question_variations(question)
            
            for variation in question_variations:
                if variation != question:  # Don't add exact duplicates
                    variations.append({
                        'question': variation,
                        'answer': answer,
                        'category': category
                    })
        
        enhanced_dataset.extend(variations)
        logger.info(f"Enhanced dataset from {len(dataset)} to {len(enhanced_dataset)} entries")
        
        return enhanced_dataset
    
    def generate_question_variations(self, question: str) -> List[str]:
        """Generate variations of a question"""
        variations = [question]
        
        # Common question transformations
        transformations = [
            # Add/remove question words
            (r'^What is ', ''),
            (r'^Where is ', ''),
            (r'^How can I ', ''),
            (r'\?$', ''),
            
            # Add common prefixes
            ('', 'Tell me about '),
            ('', 'Information about '),
            ('', 'Details about '),
        ]
        
        for old_pattern, new_pattern in transformations:
            if old_pattern:
                new_question = re.sub(old_pattern, new_pattern, question, flags=re.IGNORECASE)
            else:
                new_question = new_pattern + question
            
            if new_question != question:
                variations.append(new_question)
        
        return variations
    
    def save_dataset(self, dataset: List[Dict], output_path: str, format: str = 'json'):
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            df = pd.DataFrame(dataset)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved to {output_path} ({len(dataset)} entries)")
    
    def generate_statistics(self, dataset: List[Dict]) -> Dict:
        """Generate statistics about the dataset"""
        if not dataset:
            return {}
        
        categories = [item['category'] for item in dataset]
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        avg_question_length = sum(len(item['question']) for item in dataset) / len(dataset)
        avg_answer_length = sum(len(item['answer']) for item in dataset) / len(dataset)
        
        stats = {
            'total_pairs': len(dataset),
            'categories': category_counts,
            'avg_question_length': round(avg_question_length, 2),
            'avg_answer_length': round(avg_answer_length, 2),
        }
        
        return stats
    
    def convert_pdf_to_dataset(self, 
                              pdf_path: str, 
                              output_path: str = None,
                              format: str = 'json',
                              extraction_method: str = 'auto',
                              enhance: bool = True) -> List[Dict]:
        
        logger.info("Starting PDF to dataset conversion")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path, extraction_method)
        if not text:
            raise ValueError("Failed to extract text from PDF")
        
        # Step 2: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 3: Extract Q&A pairs
        qa_pairs = self.extract_qa_pairs(cleaned_text)
        if not qa_pairs:
            raise ValueError("No Q&A pairs found in PDF")
        
        # Step 4: Create dataset
        dataset = self.create_dataset(qa_pairs)
        
        # Step 5: Enhance dataset (optional)
        if enhance:
            dataset = self.enhance_dataset(dataset)
        
        # Step 6: Save dataset
        if output_path:
            self.save_dataset(dataset, output_path, format)
        
        # Step 7: Generate statistics
        stats = self.generate_statistics(dataset)
        logger.info(f"Dataset statistics: {stats}")
        
        return dataset

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Convert PDF to chatbot dataset')
    parser.add_argument('pdf_path', help='Path to input PDF file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('-m', '--method', choices=['auto', 'pymupdf', 'pdfplumber', 'pypdf2'],
                       default='auto', help='PDF extraction method (default: auto)')
    parser.add_argument('--no-enhance', action='store_true',
                       help='Skip dataset enhancement')
    
    args = parser.parse_args()
    
    # Create output path if not provided
    if not args.output:
        pdf_path = Path(args.pdf_path)
        args.output = pdf_path.stem + '_dataset.' + args.format
    
    # Convert PDF to dataset
    converter = PDFToDatasetConverter()
    
    try:
        dataset = converter.convert_pdf_to_dataset(
            pdf_path=args.pdf_path,
            output_path=args.output,
            format=args.format,
            extraction_method=args.method,
            enhance=not args.no_enhance
        )
        
        print(f"Successfully converted PDF to dataset!")
        print(f"Output file: {args.output}")
        print(f"Total Q&A pairs: {len(dataset)}")
        
        # Show sample entries
        print(f"\n Sample entries:")
        for i, item in enumerate(dataset[:3]):
            print(f"{i+1}. Category: {item['category']}")
            print(f"   Q: {item['question'][:80]}...")
            print(f"   A: {item['answer'][:80]}...")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Failed to convert PDF: {e}")

if __name__ == "__main__":
    main()