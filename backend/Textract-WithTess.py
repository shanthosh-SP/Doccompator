import logging
import time

# Start recording the time
start_time = time.time()
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
import nltk 
nltk.download('punkt')   
from nltk.metrics import jaccard_distance
import tika
from tika import parser
from bs4 import BeautifulSoup
import re
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    file_path = os.path.splitext(file_path)[0].strip()
    soup = BeautifulSoup(html_content, 'lxml')
    text = ' '.join(soup.stripped_strings)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    

def extract_text_from_pdf(file_path):
    # Open the scanned PDF using PyMuPDF
    pdf_document = fitz.open(file_path)
    # Define the language and page segmentation mode for Tesseract
    tesseract_config = '--psm 6 --oem 3 -l eng'
    # Initialize a variable to store the extracted text
    all_page_text = []
    # Iterate through the pages of the PDF
    for page_number, page in enumerate(pdf_document, start=1):
        # Convert the page to an image (adjust DPI as needed)
        image = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

        # Convert the PyMuPDF Pixmap to a PIL Image
        pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

        # Apply preprocessing to enhance OCR accuracy (e.g., resizing, noise reduction, thresholding)
        pil_image = pil_image.resize((pil_image.width * 3, pil_image.height * 3), Image.BILINEAR)  # Use BILINEAR resampling

        # Convert to grayscale
        pil_image = pil_image.convert('L')

        # Apply thresholding (adjust the threshold value as needed)
        threshold_value = 200
        pil_image = pil_image.point(lambda x: 0 if x < threshold_value else 255)

        # Extract text using Tesseract OCR
        page_text = pytesseract.image_to_string(pil_image,config=tesseract_config)

        # Append the extracted text to the list
        all_page_text.append(page_text)
    return all_page_text

def save_text_to_file_pdf(text, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for page_number, page_text in enumerate(text):
            page_text = page_text.replace("<p />", "")
            file.write(f"Page {page_number + 1} Text:\n{page_text}\n---\n")

def save_text_to_file_html(text,output_file):
    with open(output_html_file, 'w', encoding='utf-8') as file:
        file.write(text)

############################FOR HTML EXTRACTION##############################
input_html_file = input('Enter an HTML file path: ')
if not os.path.exists(input_html_file):
    print(f'The file {input_html_file} does not exist.')
    exit()
########################FOR PDF EXTRACTION#############################
input_pdf_file = input('Enter a PDF file path: ')
if not os.path.exists(input_pdf_file):
    print(f'The file {input_pdf_file} does not exist.')
    exit()


extracted_text_html = extract_text_from_html(input_html_file)
extracted_text_pdf = extract_text_from_pdf(input_pdf_file)
input_html_file = os.path.splitext(input_html_file)[0].strip()
input_pdf_file = os.path.splitext(input_pdf_file)[0].strip()
output_pdf_file = f"{input_pdf_file}_extracted.txt"
output_html_file = f'{input_html_file}_extracted.txt'

if extracted_text_html is not None:
    save_text_to_file_pdf(extracted_text_pdf, output_pdf_file)
    save_text_to_file_html(extracted_text_html, output_html_file)
    logging.info(f"Extracted text saved to {output_pdf_file}")
    logging.info(f"Extracted text saved to {output_html_file}")

else:
    print("Error: The number of extracted pages does not match the PDF metadata.")

##############STORING IN TEXT FILES#################
pdf_text = ""
with open(output_pdf_file, 'r', encoding='utf-8') as pdf_file:
    pdf_text = pdf_file.read().lower()

html_text = ""
with open(output_html_file, 'r', encoding='utf-8') as html_file:
    html_text = html_file.read().lower()


#############TOKENIZING WORDS############################3
# pdf_words = set(re.findall(r'\b\w+\b', pdf_text))

# html_words = set(re.findall(r'\b\w+\b', html_text))
# from nltk.tokenize import WordPunctTokenizer
     
# # Create a reference variable for Class SpaceTokenizer
# tk = WordPunctTokenizer()
# nltk.download('punkt')
pdf_words = set(nltk.word_tokenize(pdf_text))
html_words = set(nltk.word_tokenize(html_text))

#################FINDING DIFFERENCE##########################
difference_words = pdf_words - html_words 


###################FINDING THE LINE,POSITION,PAGE##############
word_positions = {}
pdf_lines = pdf_text.splitlines()

page_number = 0
line_number = 0

for line in pdf_lines:
    if line.startswith("page"):
        match = re.match(r'page (\d+) text:', line)
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
            
input_html_file = re.sub(r'[\\\/]', '_', input_html_file)
input_pdf_file = re.sub(r'[\\\/]', '_', input_pdf_file)
output = f'{input_html_file}_{input_pdf_file}_tess_Position-Line-Page.txt'
with open(output, 'w', encoding='utf-8') as result_file:
    for word, positions in word_positions.items():
        for data in positions:
            result_file.write(f"Word: {word}\n")
            result_file.write(f"Page: {data['Page']}\n")
            result_file.write(f"Line: {data['Line']}\n")
            result_file.write(f"Position: {data['Position']}\n")
            result_file.write(f"Line Content: {data['LineContent']}\n\n")

logging.info(f"Words in PDF but not in HTML, along with positions, saved to {output}")

#############COMPARSION USING JACCARD SIMILARITY######################
#import SpaceTokenizer() method from nltk

# pdf_words=set(pdf_text.split())
# html_words=set(pdf_text.split())
#print(f"OVER HERE PDF {pdf_words}")

jaccard_similarity = 1 - jaccard_distance(pdf_words, html_words)
percentage_difference = (1 - jaccard_similarity) * 100

logging.info(f"Jaccard Similarity Score: {jaccard_similarity:.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Overall execution time: {elapsed_time:.2f} seconds")