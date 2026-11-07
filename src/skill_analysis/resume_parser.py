
import pymupdf
from docx import Document
from pathlib import Path


def extract_pdf_text(file_path):
    text = ""

    pdf = pymupdf.open(file_path)

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()
    return text


def extract_docx_text(file_path):
    document = Document(file_path)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text += cell.text + " "

    return text


def extract_resume_text(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    else:
        raise ValueError("Only PDF and DOCX files are supported.")


if __name__ == "__main__":
    print("Resume parser module is ready!")
