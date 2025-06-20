import os
import PyPDF2


import fitz  # pip install PyMuPDF
from PIL import Image


def pdf_parsing(pdf_path, save_dir, dpi=300):
    os.makedirs(save_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        image.save(f"{save_dir}/{i}.png")



def merge_pdfs(input_path, output_path):
    pdf_list = os.listdir(input_path)
    pdf_list = list(sorted(pdf_list, key=lambda item: int(item.split(".")[0])))
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf = os.path.join(input_path, pdf)
        with open(pdf, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page])

    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

