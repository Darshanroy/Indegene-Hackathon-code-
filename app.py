from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
from pymongo import MongoClient
import re  # Import for re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from huggingface_hub import login
from flask_cors import CORS
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Incorrect
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # Incorrect
from langchain_cohere import CohereEmbeddings  # Correct import
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import logging  # Import logging
from faiss_utils import process_search_request, initialize_index, load_index_and_files, sanitize_filename
from rag_utils import process_pdf_rag, query_rag_chain


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration ---
# No More MongoDB Configuration

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'database/upload/')  # Get from env or default

ALLOWED_EXTENSIONS = {'pdf'}  # Only allow PDF files

# Indexing Configuration
app.config['INDEXING_FOLDER'] = os.environ.get('INDEXING_FOLDER', 'database/indexing/')  # Get from env or default

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEXING_FOLDER'], exist_ok=True)

# Hugging Face and Model Configuration
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")  # Environment Variables
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")  # Environment Variables

# Paths for FAISS index and PDF file list (relative to INDEXING_FOLDER)
INDEX_PATH = os.path.join(app.config['INDEXING_FOLDER'], "faiss_index.bin")
PDF_FILES_PATH = os.path.join(app.config['INDEXING_FOLDER'], "pdf_files.pkl")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# --- Global Variables (Initialized when app starts) ---
model = None  # Embedding model
dimension = None  # FAISS index dimension
index = None  # FAISS index
pdf_files = []  # List of PDF filenames

# --- Helper Functions ---
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def configure_huggingface_login(hf_token=None):
    """Logs in to Hugging Face Hub if a token is provided."""
    if not hf_token:
        logging.warning("Hugging Face API token not found in environment.  Functionality might be limited.")
        return False

    try:
        login(token=hf_token)
        logging.info("Successfully logged in to Hugging Face Hub.")
        return True
    except Exception as e:
        logging.exception(f"Error logging in to Hugging Face Hub: {e}")
        return False

def load_embedding_model(model_name):
    """Loads the SentenceTransformer embedding model."""
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logging.exception(f"Error loading embedding model: {e}")
        return None

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    file_paths = []  # List of files to store paths
    error_message = None  # Check for any exceptions

    if request.method == 'POST':
        if 'pdf_files' in request.files:
            files = request.files.getlist('pdf_files')  # Get a list of uploaded files
            new_files_uploaded = False  # Flag to check if new files are uploaded
            for file in files:  # Iterate through the files
                if file and allowed_file(file.filename):
                    # Secure the filename, and set file path
                    filename = sanitize_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        # Save the file
                        file.save(file_path)
                        new_files_uploaded = True
                        file_paths.append(file_path)  # Add to the displayed file list

                        logging.info(f"Saved file '{filename}' to {file_path}")

                    except Exception as e:
                        error_message = f"An error occurred while saving {filename}: {e}"
                        logging.exception(error_message)

                else:  # Error of allowed extension
                    error_message = f"File type not allowed, it must be .pdf"
                    logging.warning(error_message)

            if new_files_uploaded:
                # Rebuild the FAISS index after new files are uploaded
                try:
                    global index, pdf_files
                    index, pdf_files = initialize_index(
                        app.config['UPLOAD_FOLDER'],  # Use upload folder as source
                        INDEX_PATH,
                        PDF_FILES_PATH,
                        dimension,
                        model  # Pass model Instance
                    )
                    logging.info("FAISS index rebuilt after file upload.")
                except Exception as e:
                    error_message = f"An error occurred while rebuilding the index: {e}"
                    logging.exception(error_message)

                # Also, re-initialize the RAG chain if new files are uploaded
                try:
                    response = requests.post(url_for('process_pdf_rag_route', _external=True))
                    if response.status_code == 200:
                        logging.info("RAG chain re-initialized after file upload.")
                    else:
                        logging.error(f"Failed to re-initialize RAG chain: {response.content}")
                except Exception as e:
                    logging.exception(f"Error re-initializing RAG chain: {e}")


        else:
            error_message = "No files were uploaded."
            logging.warning(error_message)

    return render_template('index.html', file_paths=file_paths, error_message=error_message)

@app.route('/search_pdf_page')
def search_pdf_page():
    """Renders the PDF search page."""
    return render_template('search_pdf.html')

@app.route('/search_pdf', methods=['POST'])
def search_pdf():
    data = request.get_json()
    query = data.get('query')
    if not query:
        logging.warning("Query is required for search_pdf.")
        return jsonify({"error": "Query is required"}), 400

    try:
        # Pass upload_folder
        response = process_search_request(query, index, pdf_files, model, app.config['UPLOAD_FOLDER'])
        return jsonify(response)  # The processed response with filenames, download, and embed link
    except Exception as e:
        logging.exception(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during the search."}), 500

@app.route('/download/<filename>')
def download_pdf(filename):
    """Allows users to download the PDF file."""
    logging.info(f"Downloading PDF: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/view/<filename>')
def view_pdf(filename):
    """Serves the PDF to be embedded in the HTML."""
    logging.info(f"Viewing PDF: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/pdf_chat_page')
def pdf_chat_page():
    return render_template('index-2.html')

@app.route('/process_pdf_rag_route', methods=['POST'])
def process_pdf_rag_route():
    """Route to process PDF data for the RAG chain."""
    return process_pdf_rag(app)

@app.route('/query', methods=['POST'])
def query():
    """Endpoint to query the RAG chain."""
    data = request.get_json()
    user_question = data.get('question')
    session_id = data.get('session_id', 'default_session')
    return query_rag_chain(user_question, session_id, app)

# --- Initialization ---
if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Configure Hugging Face Login
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
    configure_huggingface_login(HF_TOKEN)

    # Load Embedding Model
    MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
    model = load_embedding_model(MODEL_NAME)
    if not model:
        raise RuntimeError("Failed to load embedding model. Exiting.")
    dimension = model.get_sentence_embedding_dimension()

    # Load Existing FAISS Index and PDF Files
    index, pdf_files = load_index_and_files(INDEX_PATH, PDF_FILES_PATH, dimension)

    # Initialize index if it doesn't exist and upload folder is not empty
    if (index is None or not pdf_files) and os.listdir(app.config['UPLOAD_FOLDER']):
        logging.info("Initializing index from upload folder...")
        index, pdf_files = initialize_index(
            app.config['UPLOAD_FOLDER'],
            INDEX_PATH,
            PDF_FILES_PATH,
            dimension,
            model
        )

    # Use app context for initialization
    with app.app_context():
        app.config['conversational_rag_chain'] = None
        app.config['vector_store'] = None
        app.config['chat_history_store'] = {}

    # --- Run App ---
    app.run(debug=True)