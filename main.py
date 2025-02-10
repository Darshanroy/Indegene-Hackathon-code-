from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from flask_cors import CORS  # Import CORS
import pickle  # For saving/loading FAISS index and pdf_files

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

from huggingface_hub import login

# Replace with your actual Hugging Face API token. Store in environment variable!
hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
if not hf_token:
    print("Warning: Hugging Face API token not found in environment.  Functionality might be limited.")
else:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Error logging in to Hugging Face Hub: {e}")


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define paths for saving the FAISS index and pdf_files
INDEX_PATH = "faiss_index.bin"
PDF_FILES_PATH = "pdf_files.pkl"


# Initialize FAISS index and pdf_files
dimension = 384  # Depends on embedding model
index = None  # Initialize to None
pdf_files = []  # Store filenames for reference


def load_index_and_files():
    global index, pdf_files
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(PDF_FILES_PATH, 'rb') as f:
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


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def generate_embedding(text):
    return model.encode(text, convert_to_tensor=True)


def store_embeddings(pdf_texts, filenames):
    global pdf_files, index
    embeddings = np.array([generate_embedding(text).cpu().numpy() for text in pdf_texts])
    index.add(embeddings)
    pdf_files.extend(filenames)


def retrieve_similar_pdfs(query, top_k=2):
    query_embedding = generate_embedding(query).cpu().numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [pdf_files[i] for i in indices[0]]


# Load and index PDFs on startup
def initialize_index(pdf_folder):
    global pdf_files, index
    pdf_texts, filenames = [], []
    index = faiss.IndexFlatL2(dimension)  # Re-initialize the index
    pdf_files = []  # Clear existing file list
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
        store_embeddings(pdf_texts, filenames)
        print(f"Indexed {len(pdf_files)} PDFs.")
        save_index_and_files() # Save the index and files
    else:
        print("No PDFs found or successfully processed in the specified folder.")


# Replace with your actual path.  Important: Make this configurable in a real deployment
pdf_folder = "Data-PDF"  # Or use environment variables for configuration
# Ensure the folder exists before initialization.  Otherwise, the app will fail.

load_index_and_files()  # Load index and PDF files on startup


if not os.path.exists(pdf_folder) and (index is None or not pdf_files):
    print(f"Error: PDF folder '{pdf_folder}' does not exist, and no pre-existing index found. The application will not function correctly.")

elif (index is None or not pdf_files) and os.path.exists(pdf_folder):
    print("No existing index found, initializing from PDF folder...")
    initialize_index(pdf_folder) # Build new index


@app.route('/', methods=['GET'])
def index_page():
    """Renders the main search page."""
    return render_template('index.html')  # Create index.html in a 'templates' folder



@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        similar_pdfs = retrieve_similar_pdfs(query)
        return jsonify({"results": similar_pdfs})
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during the search."}), 500


@app.route('/reindex', methods=['POST'])
def reindex():
    """
    Reindexes the PDFs in the configured folder.
    This is useful if the PDF content changes or if the application
    is started with an empty or incorrect index.  Requires a POST request
    as it's a state-changing operation.
    """
    try:
        initialize_index(pdf_folder)
        return jsonify({"message": "PDFs reindexed successfully."}), 200
    except Exception as e:
        print(f"Error during reindexing: {e}")
        return jsonify({"error": f"An error occurred during reindexing: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)