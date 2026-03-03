from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def extract_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_text(text)