# pdf_api.py
import logging
import subprocess
from tika import parser
import os
from inscriptis import get_text
# import nltk
# nltk.download('punkt') 
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify  # Import jsonify for JSON response
import nltk
from nltk import download

# Disable SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download 'punkt'
download('punkt')

def extract_text_with_layout(file_path):
    with open(file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    return text

def process_pdf_and_html(input_pdf_file, input_html_file):
    # Process PDF
    output_text_file = 'output.txt'
    subprocess.run(['pdftotext', '-layout', input_pdf_file, output_text_file])

    parsed_pdf = parser.from_file(input_pdf_file)
    num_pages = int(parsed_pdf['metadata'].get('xmpTPg:NPages', 0))

    extracted_text = extract_text_with_layout(output_text_file)
    pages = extracted_text.split('\x0c')
    pages_with_numbers = []

    for num_pages, page_text in enumerate(pages, 1):
        if page_text.strip():
            pages_with_numbers.append(f"Page {num_pages}\n{page_text}")

    combined_text = '\n'.join(pages_with_numbers)

    # Process HTML
    if not os.path.exists(input_html_file):
        print(f'The file {input_html_file} does not exist.')
        exit()

    with open(input_html_file, 'r', encoding="ISO-8859-1") as file:
        html_content = file.read()

    # Assuming you have a function get_text() to extract text from HTML
    text_content = get_text(html_content)

    return combined_text, text_content

def compare_pdf_with_html(pdf_text, html_text):
    try:
        from nltk.tokenize import regexp_tokenize 
        pdf_words = set(regexp_tokenize(pdf_text,"[\w']+"))
        html_words = set(regexp_tokenize(html_text, "[\w']+"))

        # Finding the difference
        difference_words = pdf_words - html_words

        # Finding the line, position, page
        word_positions = {}
        pdf_lines = pdf_text.splitlines()

        page_number = 0
        line_number = 0

        for line in pdf_lines:
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

        output = f"pdf_htmlcompare.txt"
        with open(output, 'w', encoding='utf-8') as result_file:
            for word, positions in word_positions.items():
                for data in positions:
                    result_file.write(f"Word: {word}\n")
                    result_file.write(f"Page: {data['Page']}\n")
                    result_file.write(f"Line: {data['Line']}\n")
                    result_file.write(f"Position: {data['Position']}\n")
                    result_file.write(f"Line Content: {data['LineContent']}\n\n")

        logging.info(f"Words in PDF but not in HTML, along with positions, saved to {output}")

        # Comparison using Jaccard Similarity
        jaccard_similarity = len(pdf_words.intersection(html_words)) / len(pdf_words.union(html_words))
        percentage_difference = (1 - jaccard_similarity) * 100
        logging.info(f"Jaccard Similarity Score: {jaccard_similarity:.2f}")

        # Comparison using BERT Cosine Similarity
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        tokens1 = tokenizer(pdf_text, return_tensors='pt', padding=True, truncation=True)
        tokens2 = tokenizer(html_text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
            embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

        similarity = cosine_similarity(embeddings1, embeddings2)
        bert_cosine_similarity = similarity[0][0].item()  # Convert float32 to a standard Python float
        logging.info(f"Bert Cosine Similarity: {bert_cosine_similarity:.2f}")

        # Include extracted PDF and HTML texts and comparison output in the response
        with open(output, 'r', encoding='utf-8') as result_file:
            output_content = result_file.read()

        response_data = {
            "bert_cosine_similarity": float(similarity[0][0]),
            "jaccard_similarity": float(jaccard_similarity),
            "pdf_text": pdf_text,
            "html_text": html_text,
            "comparison_output": {
                "file_path": output,
                "content": output_content
            }
        }

        return response_data
    
    except Exception as e:
        return {"error": str(e)}

# ... (other functions if any)
