import os
import openai
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# Initialize OpenAI API
openai.api_key = "your_openai_api_key"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = Client(Settings(persist_directory="vector_db"))

# Function to process PDF files and extract text
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to create or update Chroma DB
def create_vector_store(pdf_text, doc_id):
    # Define embedding function for Chroma
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model=embedding_model)
    collection = chroma_client.create_collection("pdf_documents", embedding_function=embed_fn)
    
    # Split the text into smaller chunks
    chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    return f"Added {len(chunks)} chunks to the vector store."

# Query the Chroma DB
def query_vector_store(query, top_k=3):
    collection = chroma_client.get_collection("pdf_documents")
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"]

# Generate a response from OpenAI
def generate_response(context, query):
    prompt = f"""
    Use the following context to answer the question:
    {context}
    
    Question: {query}
    Answer:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
