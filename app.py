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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration ---
# MongoDB Configuration
app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')  # Get from env or default
app.config['DATABASE_NAME'] = os.environ.get('DATABASE_NAME', 'pdf_app_db')

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


# Initialize MongoDB
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['DATABASE_NAME']]
pdf_files_collection = db['pdf_files']  # Collection for individual files

# --- Global Variables (Initialized when app starts) ---
model = None  # Embedding model
dimension = None  # FAISS index dimension
index = None  # FAISS index
pdf_files = []  # List of PDF filenames


# --- Helper Functions ---
def sanitize_filename(filename):
    """
    Sanitizes a filename by removing or replacing potentially dangerous characters.
    This is a simplified and potentially less secure approach.
    """
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)  # Replace non-alphanumeric with underscores
    ext = re.sub(r'[^a-zA-Z0-9\.]', '', ext)      # Allow only alphanumeric and dots in extension
    return name + ext


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def configure_huggingface_login(hf_token=None):
    """Logs in to Hugging Face Hub if a token is provided."""
    if not hf_token:
        print("Warning: Hugging Face API token not found in environment.  Functionality might be limited.")
        return False

    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub.")
        return True
    except Exception as e:
        print(f"Error logging in to Hugging Face Hub: {e}")
        return False


def load_embedding_model(model_name):
    """Loads the SentenceTransformer embedding model."""
    try:
        model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None


def load_index_and_files(index_path, pdf_files_path, dimension):
    """Loads the FAISS index and PDF file list from disk."""
    index = None
    pdf_files = []

    try:
        index = faiss.read_index(index_path)
        with open(pdf_files_path, 'rb') as f:
            pdf_files = pickle.load(f)
        print("Loaded FAISS index and PDF files from disk.")
    except FileNotFoundError:
        print("FAISS index or PDF files not found. Creating a new index.")
        index = faiss.IndexFlatL2(dimension)  # Create a new index
        pdf_files = []  # Start with an empty list of files
    except Exception as e:
        print(f"Error loading index or PDF files: {e}. Creating a new index.")
        index = faiss.IndexFlatL2(dimension)  # Create a new index
        pdf_files = []  # Start with an empty list of files

    return index, pdf_files


def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def generate_embedding(text, model_instance):  # Use model_instance
    """Generates an embedding vector for the given text."""
    return model_instance.encode(text, convert_to_tensor=True)


def store_embeddings(pdf_texts, filenames, index_instance, pdf_files_list, model_instance):  # Use instance
    """Stores embeddings in the FAISS index."""
    embeddings = np.array([generate_embedding(text, model_instance).cpu().numpy() for text in pdf_texts])
    index_instance.add(embeddings)
    pdf_files_list.extend(filenames)
    return index_instance, pdf_files_list


def save_index_and_files(index_instance, pdf_files_list, index_path, pdf_files_path):
    """Saves the FAISS index and PDF file list to disk."""
    try:
        faiss.write_index(index_instance, index_path)
        with open(pdf_files_path, 'wb') as f:
            pickle.dump(pdf_files_list, f)
        print("FAISS index and PDF files saved to disk.")
    except Exception as e:
        print(f"Error saving index and PDF files: {e}")


def initialize_index(pdf_folder, index_path, pdf_files_path, dimension_value, model_instance):  # Add model instance
    """Initializes the FAISS index by processing PDFs in a folder."""
    pdf_texts, filenames = [], []
    index_instance = faiss.IndexFlatL2(dimension_value)  # Re-initialize the index
    pdf_files_list = []  # Clear existing file list
    print(f"Loading PDFs from: {pdf_folder}")

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, file)
            print(f"Processing {filepath}...")
            text = extract_text_from_pdf(filepath)
            if text:  # Only append if text extraction was successful
                pdf_texts.append(text)
                filenames.append(file)
            else:
                print(f"Skipping {file} due to text extraction failure.")

    if pdf_texts:  # Only store embeddings if there are any PDFs loaded
        index_instance, pdf_files_list = store_embeddings(pdf_texts, filenames, index_instance, pdf_files_list,
                                                          model_instance)
        print(f"Indexed {len(pdf_files_list)} PDFs.")
        save_index_and_files(index_instance, pdf_files_list, index_path, pdf_files_path)
    else:
        print("No PDFs found or successfully processed in the specified folder.")

    return index_instance, pdf_files_list


def retrieve_similar_pdfs(query, index_instance, pdf_files_list, model_instance, top_k=2):
    """Retrieves similar PDFs based on the query."""
    query_embedding = generate_embedding(query, model_instance).cpu().numpy().reshape(1, -1)
    distances, indices = index_instance.search(query_embedding, top_k)
    return [pdf_files_list[i] for i in indices[0]]


def process_search_request(query, index_instance, pdf_files_list, model_instance, upload_folder):
    """
    Processes a search request, retrieves similar PDFs, and returns:
    - Filename
    - Download Link
    - Embed link
    """

    # Extract Query from Request
    print(f"Query received: {query}")

    # Generate Embedding for Query
    try:
        similar_pdfs = retrieve_similar_pdfs(query, index_instance, pdf_files_list, model_instance)
    except Exception as e:
        print(f"Error during search: {e}")
        return {"error": "An error occurred during the search."}

    # Create response with filename, download link, and embed link for each PDF
    results = []
    for filename in similar_pdfs:
        download_link = url_for('download_pdf', filename=filename, _external=True)  # Generates the fully qualified URL
        embed_link = url_for('view_pdf', filename=filename, _external=True)  # Create an URL endpoint to render the PDF
        results.append({
            "filename": filename,
            "download_link": download_link,
            "embed_link": embed_link
        })

    print(f"Found similar PDFs: {results}")
    return {"results": results}


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
                    # Secure the filename, and set file path in database
                    filename = sanitize_filename(file.filename)  # Use sanitize_filename
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        # Save the file
                        file.save(file_path)
                        new_files_uploaded = True

                        # Save file path to MongoDB
                        file_data = {
                            'filename': filename,
                            'file_path': file_path
                        }
                        result = pdf_files_collection.insert_one(file_data)
                        print(f"Saved file '{filename}' to MongoDB with ID: {result.inserted_id}")
                        file_paths.append(file_path)

                    except Exception as e:
                        error_message = f"An error occurred while saving {filename}: {e}"
                        print(error_message)

                else:  # Error of allowed extension
                    error_message = f"File type not allowed, it must be .pdf"

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
                    print("FAISS index rebuilt after file upload.")
                except Exception as e:
                    error_message = f"An error occurred while rebuilding the index: {e}"
                    print(error_message)

        else:
            error_message = "No files were uploaded."

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
        return jsonify({"error": "Query is required"}), 400

    try:
        # Pass upload_folder
        response = process_search_request(query, index, pdf_files, model, app.config['UPLOAD_FOLDER'])
        return jsonify(response)  # The processed response with filenames, download, and embed link
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during the search."}), 500


@app.route('/download/<filename>')
def download_pdf(filename):
    """Allows users to download the PDF file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/view/<filename>')
def view_pdf(filename):
    """Serves the PDF to be embedded in the HTML."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Initialization ---
if __name__ == '__main__':
    # Configure Hugging Face Login
    configure_huggingface_login(HF_TOKEN)

    # Load Embedding Model
    model = load_embedding_model(MODEL_NAME)
    if not model:
        raise RuntimeError("Failed to load embedding model.  Exiting.")
    dimension = model.get_sentence_embedding_dimension()

    # Load Existing FAISS Index and PDF Files
    index, pdf_files = load_index_and_files(INDEX_PATH, PDF_FILES_PATH, dimension)

    # Initialize index if it doesn't exist and upload folder is not empty
    if (index is None or not pdf_files) and os.listdir(app.config['UPLOAD_FOLDER']):
        print("Initializing index from upload folder...")
        index, pdf_files = initialize_index(
            app.config['UPLOAD_FOLDER'],
            INDEX_PATH,
            PDF_FILES_PATH,
            dimension,
            model  # Pass model instance
        )

    # --- Run App ---
    app.run(debug=True)