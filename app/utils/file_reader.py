import pdfplumber
from docx import Document
import io
from bs4 import BeautifulSoup


def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")
    if not text.strip():
        raise ValueError(
            "PDF appears to be empty or contains no extractable text.")
    return text.strip()


def extract_text_from_html(file):
    """Extract text from an HTML file."""
    try:
        html_content = file.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        raise ValueError(f"Error extracting text from HTML: {str(e)}")
    if not text.strip():
        raise ValueError(
            "HTML appears to be empty or contains no extractable text.")
    return text.strip()


def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    try:
        doc = Document(file)
        text = ' '.join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")
    if not text.strip():
        raise ValueError(
            "DOCX appears to be empty or contains no extractable text.")
    return text.strip()


def extract_text(file):
    """Extract text based on file type."""
    filename = getattr(file, 'name', 'unknown').lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        return extract_text_from_docx(file)
    elif filename.endswith('.html') or filename.endswith('.htm'):
        return extract_text_from_html(file)
    else:
        raise ValueError(
            "Unsupported file type. Only PDF, DOCX, and HTML are supported.")
