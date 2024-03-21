import logging
import time

# Start recording the time
start_time = time.time()
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
import nltk    
# nltk.download('punkt')   
from nltk.metrics import jaccard_distance
import tika
from tika import parser
from bs4 import BeautifulSoup
import re
import os

############################FOR HTML EXTRACTION##############################
input_html_file = input('Enter an HTML file path: ')

if not os.path.exists(input_html_file):
    print(f'The file {input_html_file} does not exist.')
    exit()

with open(input_html_file, 'r', encoding='utf-8') as file:
    html_content = file.read()

input_html_file = os.path.splitext(input_html_file)[0].strip()
soup = BeautifulSoup(html_content, 'lxml')
text = ' '.join(soup.stripped_strings)
text = re.sub(r'\s+', ' ', text).strip()

output_html_file = f'{input_html_file}_extracted.txt'

with open(output_html_file, 'w', encoding='utf-8') as file:
    file.write(text)
logging.info(f"Extracted and cleaned text saved to {output_html_file}")

########################FOR PDF EXTRACTION#############################
input_pdf_file = input('Enter a PDF file path: ')

if not os.path.exists(input_pdf_file):
    print(f'The file {input_pdf_file} does not exist.')
    exit()

def extract_text_from_pdf(file_path):
    raw_xml = parser.from_file(file_path, xmlContent=True)
    body = raw_xml['content'].split('<body>')[1].split('</body>')[0]
    
    text_pages = body.split('<div class="page">')[1:]
    
    num_pages = len(text_pages)
    if num_pages == int(raw_xml['metadata']['xmpTPg:NPages']):
        cleaned_pages = [page.replace("<p>", "").replace("</p>", "").replace("<div>", "").replace("</div>", "").strip() for page in text_pages]
        return cleaned_pages
    else:
        return None

def save_text_to_file(text, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for page_number, page_text in enumerate(text):
            page_text = page_text.replace("<p />", "")
            file.write(f"Page {page_number + 1} Text:\n{page_text}\n---\n")

extracted_text = extract_text_from_pdf(input_pdf_file)
input_pdf_file = os.path.splitext(input_pdf_file)[0].strip()
output_pdf_file = f"{input_pdf_file}_extracted.txt"

if extracted_text is not None:
    save_text_to_file(extracted_text, output_pdf_file)
    logging.info(f"Extracted text saved to {output_pdf_file}")

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
from nltk.tokenize import regexp_tokenize 
pdf_words = set(regexp_tokenize(pdf_text,"[\w']+"))
html_words = set(regexp_tokenize(html_text, "[\w']+"))

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
output = f'{input_html_file}_{input_pdf_file}_Position-Line-Page.txt'
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