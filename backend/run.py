# run.py
import os
from flask import Flask, request, jsonify
#from docx_api import extract_text_from_docx, compare_docx_with_html
from pdf_api import process_pdf_and_excel, compare_pdf_with_html
from ppt_api import extract_text_from_ppt, compare_ppt_with_html  # Import new functions
from excel_api import process_files_and_return_results 
app = Flask(__name__)

@app.route('/process_and_compare_files', methods=['POST'])
def process_and_compare_files():
    try:
        file1 = request.files['file1']
        html_file = request.files['html_file']

        # Get the file paths from the request
        file1_filename, html_filename = file1.filename, html_file.filename
        file1_extension = os.path.splitext(file1_filename)[1].lower()
        html_extension = os.path.splitext(html_filename)[1].lower()

        if file1_extension == '.pdf' and html_extension == '.xlsx':
            # Process PDF and HTML files
            pdf_file_path = 'temp_pdf.pdf'
            excel_file_path1 = 'temp_htl.xlsx'
            file1.save(pdf_file_path)
            html_file.save(excel_file_path1)
            print("#############")

            pdf_text, excel_text = process_pdf_and_excel(pdf_file_path, excel_file_path1)
            print("########333333333333#####")

            # Compare PDF with HTML
            pdf_html_comparison_result = compare_pdf_with_html(pdf_text, excel_text)

            return jsonify(pdf_html_comparison_result)
        elif file1_extension == '.xlsx' and html_extension == '.html':
            # Save the files to temporary locations
            excel_file_path = 'temp_excel.xlsx'
            html_file_path = 'temp_html.html'
            file1.save(excel_file_path)
            html_file.save(html_file_path)
 
            # Compare the texts using the function from pdf_api.py
            comparison_result = process_files_and_return_results(excel_file_path,html_file_path)
            return jsonify(comparison_result)
        # elif file1_extension == '.docx' and html_extension == '.html':
        #     # Process DOCX and HTML files
        #     docx_file_path = 'temp_docx.docx'
        #     html_file_path = 'temp_html.html'
        #     file1.save(docx_file_path)
        #     html_file.save(html_file_path)

        #     docx_text, html_text = extract_text_from_docx(docx_file_path, html_file_path)

        #     # Compare DOCX with HTML
        #     docx_html_comparison_result = compare_docx_with_html(docx_text, html_text)

        #     return jsonify(docx_html_comparison_result)

        elif file1_extension == '.pptx' and html_extension == '.html':
            # Process PPTX and HTML files
            pptx_file_path = 'temp_ppt.pptx'
            html_file_path = 'temp_html.html'
            file1.save(pptx_file_path)
            html_file.save(html_file_path)

            ppt_text,html_text = extract_text_from_ppt(pptx_file_path, html_file_path)

            # Compare PPTX with HTML
            ppt_html_comparison_result = compare_ppt_with_html(ppt_text, html_text)

            return jsonify(ppt_html_comparison_result)

        else:
            return jsonify({"error": "Unsupported file format combination. Please provide a PDF and an HTML file, a DOCX and an HTML file, or a PPTX and an HTML file."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
