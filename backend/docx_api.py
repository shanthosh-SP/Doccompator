# docx_api.py

from docx import Document
from docx2pdf import convert
from weasyprint import HTML
from tika import parser
from inscriptis import get_text
import os
import logging
import subprocess
import nltk
from nltk import download
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
#from reportlab.pdfgen import canvas
#nltk.download('punkt') 

# Set the data download path
nltk.data.path.append("D:\comparison_tool\documentcomparisiontool")

# Disable SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download 'punkt'
download('punkt')

def convert_docx_to_pdf(docx_path, pdf_path):
    try:
        # Use docx2pdf library for conversion
        convert(docx_path, pdf_path)
        print(f"Conversion successful: {docx_path} -> {pdf_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")

def extract_text_from_docx(input_docs_file, input_html_file):
    pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Temp_pdf.pdf')

    try:
        print("Converting DOCX to PDF...")
        convert_docx_to_pdf(input_docs_file, pdf_path)
        print(f"Conversion successful. PDF saved to {pdf_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

    # Assuming you have a function get_text() to extract text from HTML
    with open(input_html_file, 'r', encoding="ISO-8859-1") as file:
        html_content = file.read()

    text_content = get_text(html_content)

    # Extract text from PDF with layout preservation
    input_pdf_file = 'Temp_pdf.pdf'
    output_text_file = 'tempext.txt'

    # Use pdftotext with layout preservation
    subprocess.run(['pdftotext', '-layout', input_pdf_file, output_text_file])

    # Use Tika to extract page numbers from PDF metadata
    parsed_pdf = parser.from_file(input_pdf_file)
    num_pages = int(parsed_pdf['metadata'].get('xmpTPg:NPages', 0))

    # Function to extract text with layout preservation
    def extract_text_with_layout(file_path):
        with open(file_path, 'r', encoding='utf-8') as text_file:
            text = text_file.read()
        return text

    # Read the extracted text with layout preservation
    extracted_text = extract_text_with_layout(output_text_file)

    # Split the text into pages based on page breaks
    pages = extracted_text.split('\x0c')

    # Create a list to store the page numbers along with their corresponding text
    pages_with_numbers = []

    for page_number, page_text in enumerate(pages, 1):
        if page_text.strip():  # Check if the page is not empty or mostly empty
            pages_with_numbers.append(f"Page {page_number}\n{page_text}")

    # Combine all pages
    combined_text = '\n'.join(pages_with_numbers)

    # Create a new text file that combines page numbers and text
    input_temptext_file = os.path.splitext(input_docs_file)[0].strip()
    output_temptext_file = f'{input_temptext_file}_temptext_extracted.txt'

    with open(output_temptext_file, 'w', encoding='utf-8') as combined_text_file:
        combined_text_file.write(combined_text)

    # Now you have a single text file containing page numbers along with their corresponding text content and layout
    print(f"Combined text with page numbers and layout preserved saved to {output_temptext_file}")

    return combined_text, text_content

# Rest of your code remains unchanged


def compare_docx_with_html(docx_text, html_text):
    # Tokenize the text
    docs_words = set(nltk.word_tokenize(docx_text))
    html_words = set(nltk.word_tokenize(html_text))

    # Finding the difference
    difference_words = docs_words - html_words

    # Finding the line, position, page
    word_positions = {}
    docs_lines = docx_text.splitlines()

    page_number = 0
    line_number = 0

    for line in docs_lines:
        if line.startswith("Page"):
            match = re.match(r'Page (\d+)', line)
            if match:
                page_number = int(match.group(1))
            line_number = 0
        else:
            line_number += 1

        line_words = list(re.findall(r'\b\w+\b', line))
        for word in line_words:
            if word in difference_words:
                if word not in word_positions:
                    word_positions[word] = []
                word_positions[word].append({
                    'Page': page_number,
                    'Line': line_number,
                    'Position': line_words.index(word),
                    'LineContent': line
                })

    output = f'docs_htmlcompare.txt'
    with open(output, 'w', encoding='utf-8') as result_file:
        for word, positions in word_positions.items():
            for data in positions:
                result_file.write(f"Word: {word}\n")
                result_file.write(f"Page: {data['Page']}\n")
                result_file.write(f"Line: {data['Line']}\n")
                result_file.write(f"Position: {data['Position']}\n")
                result_file.write(f"Line Content: {data['LineContent']}\n\n")

    logging.info(f"Words in Docx but not in HTML, along with positions, saved to {output}")

    # Comparison using Jaccard Similarity
    jaccard_similarity = len(docs_words.intersection(html_words)) / len(docs_words.union(html_words))
    percentage_difference = (1 - jaccard_similarity) * 100
    logging.info(f"Jaccard Similarity Score: {jaccard_similarity:.2f}")

    # Comparison using BERT Cosine Similarity
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    tokens1 = tokenizer(docx_text, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(html_text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(embeddings1, embeddings2)
    logging.info(f"Bert Cosine Similarity: {similarity[0][0]:.2f}")

    with open(output, 'r', encoding='utf-8') as result_file:
            output_content = result_file.read()

    return {
        "bert_cosine_similarity": float(similarity[0][0]),
        "jaccard_similarity": float(jaccard_similarity),
        "docx_text": docx_text,
        "html_text": html_text,
        "comparison_output": {
            "file_path": output,
            "content": output_content  # You can include the content if needed
        }
    }
