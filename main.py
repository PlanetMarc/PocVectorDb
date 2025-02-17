import os
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

class MyEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    def __call__(self, input):
        return self.model.encode(input).tolist()

def main():
    # Initialize the embedding model using SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_function = MyEmbeddingFunction(model)
    
    # Initialize Chroma DB client with DuckDB storage
    client = chromadb.Client(Settings(persist_directory="./db"))
    
    # Attempt to get existing collection or create a new one
    try:
        collection = client.get_collection("pdfs")
    except Exception:
        collection = client.create_collection("pdfs", embedding_function=embedding_function)
    
    pdf_directory = "pdfs"
    if not os.path.exists(pdf_directory):
        print("PDF directory not found. Please create a 'pdfs' folder and add PDF files to ingest.")
        return
    
    # Ingest each PDF file from the 'pdfs' directory
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(file_path)
            if text.strip():
                collection.add(
                    documents=[text],
                    ids=[filename],
                    metadatas=[{"filename": filename}],
                )
                print(f"Ingested {filename}")
            else:
                print(f"No extractable text found in {filename}")

if __name__ == "__main__":
    import sys
    if "--skip-model" in sys.argv:
        print("Skipping model load as per argument.")
    else:
        main()