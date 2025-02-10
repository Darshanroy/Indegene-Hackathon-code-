import os
import json
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
import pymongo
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bertopic import BERTopic

# Configuration Constants
UPLOAD_FOLDER = "./pdfs"
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "pdf_db"
COLLECTION_NAME = "pdf_metadata"
FAISS_INDEX_PATH = "pdf_index.faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "mixtral-8x7b"
GROQ_API_KEY = "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"  #Replace with actual API Key
SUMMARY_LENGTH = 500

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database Setup
def get_database_connection():
    client = pymongo.MongoClient(MONGODB_URI)
    return client[DATABASE_NAME]

db = get_database_connection()
collection = db[COLLECTION_NAME]


# Model Initialization
def initialize_models():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    topic_model = BERTopic()
    groq_llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY)
    return embedding_model, topic_model, groq_llm

embedding_model, topic_model, groq_llm = initialize_models()


# PDF Processing Functions
def extract_text_from_pdf(pdf_path):
    """Extracts text and page count from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text("text") for page in doc])
        return text, len(doc)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "", 0


def generate_embedding(text):
    """Generates embeddings for the given text using the pre-loaded embedding model."""
    try:
        return embedding_model.embed_documents([text])[0]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def generate_topics(text):
    """Generates topics for the given text using the pre-loaded topic model."""
    try:
        topics, _ = topic_model.fit_transform([text])
        return topics
    except Exception as e:
        print(f"Error generating topics: {e}")
        return []


def store_pdf_metadata(filename, topics, summary, page_count, embedding):
    """Stores PDF metadata in MongoDB."""
    try:
        doc = {
            "filename": filename,
            "topics": topics,
            "summary": summary,
            "page_count": page_count,
            # "embedding": embedding.tolist() if embedding is not None else None  # Convert to list for JSON storage
        }
        collection.insert_one(doc)
    except Exception as e:
        print(f"Error storing PDF metadata in MongoDB: {e}")


# FAISS Indexing Functions
def build_faiss_index(embeddings):
    """Builds a FAISS index from the given embeddings."""
    try:
        d = len(embeddings[0])
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings))
        faiss.write_index(index, FAISS_INDEX_PATH)
        return index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None


def load_faiss_index():
    """Loads the FAISS index from the saved file."""
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


# Helper function to process a single PDF
def process_pdf(pdf_path, all_topics):
    """Processes a single PDF file and stores its metadata."""
    filename = os.path.basename(pdf_path)
    text, page_count = extract_text_from_pdf(pdf_path)
    if not text:
        return None  # Skip if text extraction failed

    embedding = generate_embedding(text)
    summary = text[:SUMMARY_LENGTH]

    store_pdf_metadata(filename, all_topics, summary, page_count, embedding)

    return embedding  # Return embedding for index building


# Route Handlers
@app.route('/')
def serve_frontend():
    """Serves the frontend application."""
    return send_from_directory("templates", "index.html")


@app.route('/upload', methods=['POST'])
def upload_pdfs():
    """Handles PDF file uploads."""
    if 'pdfs' not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist('pdfs')
    uploaded_files = []
    for file in files:
        if file.filename != '':  # Check if a file was selected
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(file_path)
                uploaded_files.append(file_path)
            except Exception as e:
                return jsonify({"message": f"Error saving file {file.filename}: {e}"}), 500
    
    if not uploaded_files:
        return jsonify({"message": "No valid files uploaded."}), 400

    return jsonify({"message": "Files uploaded successfully!"}), 200


@app.route('/process', methods=['POST'])
def process():
    """Processes uploaded PDFs, generates embeddings, and builds the FAISS index."""
    pdf_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(".pdf")]
    all_embeddings = []
    all_text = []  # Store text from all PDFs

    for pdf_path in pdf_files:
        text, page_count = extract_text_from_pdf(pdf_path)
        if not text:
            continue  # Skip if text extraction failed
        all_text.append(text)

    if all_text:
        combined_text = " ".join(all_text) # Concatenate the texts
        topics = generate_topics(combined_text) # Now generate topics from all texts

        # Generate embeddings and store metadata for each PDF based on combined topics:
        for pdf_path in pdf_files:
          embedding = process_pdf(pdf_path, topics)
          if embedding is not None:
              all_embeddings.append(embedding)

        build_faiss_index(all_embeddings)
        return jsonify({"message": "Processing completed!"}), 200
    else:
        return jsonify({"message": "No PDFs processed. Ensure PDFs are valid."}), 500


@app.route('/search', methods=['POST'])
def search():
    """Searches for PDFs based on a query using FAISS index."""
    data = request.get_json()
    query = data.get("query", "")

    index = load_faiss_index()
    if index is None:
        return jsonify({"message": "FAISS index not found. Please process PDFs first."}), 500

    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return jsonify({"message": "Error generating embedding for the query."}), 500

    query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape for FAISS
    _, idxs = index.search(query_embedding, 3)  # Search top 3

    results = []
    for i, doc in enumerate(collection.find()):
        if i in idxs[0]:
            results.append(doc)

    return jsonify(results), 200


# Main execution
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)