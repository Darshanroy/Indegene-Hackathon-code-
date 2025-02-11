# faiss_utils.py
import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import logging  # Import logging
from flask import url_for

def sanitize_filename(filename):
    """Sanitizes a filename."""
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    ext = re.sub(r'[^a-zA-Z0-9\.]', '', ext)
    return name + ext

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        logging.exception(f"Error extracting text from {pdf_path}: {e}")
        return ""

def generate_embedding(text, model_instance):
    """Generates an embedding vector."""
    return model_instance.encode(text, convert_to_tensor=True)

def store_embeddings(pdf_texts, filenames, index_instance, pdf_files_list, model_instance):
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
        logging.info("FAISS index and PDF files saved to disk.")
    except Exception as e:
        logging.exception(f"Error saving index and PDF files: {e}")

def load_index_and_files(index_path, pdf_files_path, dimension):
    """Loads the FAISS index and PDF file list from disk."""
    index = None
    pdf_files = []

    try:
        index = faiss.read_index(index_path)
        with open(pdf_files_path, 'rb') as f:
            pdf_files = pickle.load(f)
        logging.info("Loaded FAISS index and PDF files from disk.")
    except FileNotFoundError:
        logging.warning("FAISS index or PDF files not found. Creating a new index.")
        index = faiss.IndexFlatL2(dimension)  # Create a new index
        pdf_files = []  # Start with an empty list of files
    except Exception as e:
        logging.exception(f"Error loading index or PDF files: {e}. Creating a new index.")
        index = faiss.IndexFlatL2(dimension)  # Create a new index
        pdf_files = []  # Start with an empty list of files

    return index, pdf_files


def initialize_index(pdf_folder, index_path, pdf_files_path, dimension_value, model_instance):
    """Initializes the FAISS index by processing PDFs in a folder."""
    pdf_texts, filenames = [], []
    index_instance = faiss.IndexFlatL2(dimension_value)  # Re-initialize the index
    pdf_files_list = []  # Clear existing file list
    logging.info(f"Loading PDFs from: {pdf_folder}")

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, file)
            logging.info(f"Processing {filepath}...")
            text = extract_text_from_pdf(filepath)
            if text:  # Only append if text extraction was successful
                pdf_texts.append(text)
                filenames.append(file)
            else:
                logging.warning(f"Skipping {file} due to text extraction failure.")

    if pdf_texts:  # Only store embeddings if there are any PDFs loaded
        index_instance, pdf_files_list = store_embeddings(pdf_texts, filenames, index_instance, pdf_files_list,
                                                          model_instance)
        logging.info(f"Indexed {len(pdf_files_list)} PDFs.")
        save_index_and_files(index_instance, pdf_files_list, index_path, pdf_files_path)
    else:
        logging.warning("No PDFs found or successfully processed in the specified folder.")

    return index_instance, pdf_files_list

def retrieve_similar_pdfs(query, index_instance, pdf_files_list, model_instance, top_k=2):
    """Retrieves similar PDFs based on the query."""
    query_embedding = generate_embedding(query, model_instance).cpu().numpy().reshape(1, -1)
    distances, indices = index_instance.search(query_embedding, top_k)
    return [pdf_files_list[i] for i in indices[0]]

def process_search_request(query, index_instance, pdf_files_list, model_instance, upload_folder):
    """Processes a search request, retrieves similar PDFs."""
    # Extract Query from Request
    logging.info(f"Query received: {query}")

    # Generate Embedding for Query
    try:
        similar_pdfs = retrieve_similar_pdfs(query, index_instance, pdf_files_list, model_instance)
    except Exception as e:
        logging.exception(f"Error during search: {e}")
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

    logging.info(f"Found similar PDFs: {results}")
    return {"results": results}