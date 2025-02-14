# faiss_utils.py
import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import logging
from flask import url_for, jsonify
from rag_utils import initialize_vector_store, create_conversational_rag_chain, load_documents_from_directory, CHROMA_DB_PATH  # Import necessary functions
from langchain_groq import ChatGroq # Import
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def sanitize_filename(filename):
    """Sanitizes a filename."""
    try:
        name, ext = os.path.splitext(filename)
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        ext = re.sub(r'[^a-zA-Z0-9\.]', '', ext)
        sanitized_filename = name + ext
        logging.info(f"Sanitized filename from {filename} to {sanitized_filename}")
        return sanitized_filename
    except Exception as e:
        logging.exception(f"Error sanitizing filename {filename}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()  # Close the document to release resources
        logging.info(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        logging.exception(f"Error extracting text from {pdf_path}: {e}")
        return ""

def generate_embedding(text, model_instance):
    """Generates an embedding vector."""
    try:
        embedding = model_instance.encode(text, convert_to_tensor=True)
        logging.debug(f"Generated embedding for text: {text[:50]}...")  # Log first 50 characters
        return embedding
    except Exception as e:
        logging.exception(f"Error generating embedding for text: {e}")
        return None

def store_embeddings(pdf_texts, filenames, index_instance, pdf_files_list, model_instance):
    """Stores embeddings in the FAISS index."""
    try:
        embeddings = np.array([generate_embedding(text, model_instance).cpu().numpy() for text in pdf_texts if generate_embedding(text, model_instance) is not None])  # Filter out None embeddings
        valid_filenames = [filename for text, filename in zip(pdf_texts, filenames) if generate_embedding(text, model_instance) is not None]  # Filter filenames corresponding to valid embeddings

        if embeddings.size > 0:
            index_instance.add(embeddings)
            pdf_files_list.extend(valid_filenames)
            logging.info(f"Stored {embeddings.shape[0]} embeddings in FAISS index.")
        else:
            logging.warning("No valid embeddings to store.")
        return index_instance, pdf_files_list
    except Exception as e:
        logging.exception(f"Error storing embeddings: {e}")
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
    try:
        query_embedding = generate_embedding(query, model_instance)
        if query_embedding is None:
            logging.warning("Could not generate embedding for query, returning empty list.")
            return []
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
        distances, indices = index_instance.search(query_embedding, top_k)
        similar_pdfs = [pdf_files_list[i] for i in indices[0]]
        logging.info(f"Retrieved similar PDFs for query '{query[:50]}...': {similar_pdfs}")  # Log first 50 characters of query
        return similar_pdfs
    except Exception as e:
        logging.exception(f"Error retrieving similar PDFs: {e}")
        return []


def get_or_create_pdf_rag_chain(app, filename, upload_folder):
    """Gets or creates a RAG chain for a specific PDF."""
    try:
        # Create a unique key for this PDF's RAG chain, including the filename
        rag_chain_key = f"rag_chain_{filename}"

        # Try to retrieve the RAG chain from the app's config
        if rag_chain_key in app.config:
            logging.info(f"Retrieving existing RAG chain for {filename}")
            return app.config[rag_chain_key]

        # If not found, create a new RAG chain and store it
        logging.info(f"Creating new RAG chain for {filename}")
        pdf_path = os.path.join(upload_folder, filename)
        documents = load_documents_from_directory(pdf_path)  # Load single PDF

        if not documents:
             logging.error(f"Failed to load document: {filename}")
             return None
        vector_store = initialize_vector_store(documents)  # Vector store for single PDF
        if not vector_store:
            logging.error(f"Failed to initialize vector store for {filename}")
            return None
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
        conversational_rag_chain = create_conversational_rag_chain(llm, vector_store, app)

        if conversational_rag_chain is None:
            logging.error(f"Failed to create RAG chain for {filename}")
            return None
        app.config[rag_chain_key] = conversational_rag_chain
        return conversational_rag_chain
    except Exception as e:
        logging.exception(f"Error in get_or_create_pdf_rag_chain for {filename}: {e}")
        return None

def process_search_request(query, index_instance, pdf_files_list, model_instance, upload_folder, app): #Added app
    """Processes a search request and prepares for per-document chat."""
    logging.info(f"Query received: {query}")
    try:
        similar_pdfs = retrieve_similar_pdfs(query, index_instance, pdf_files_list, model_instance)
    except Exception as e:
        logging.exception(f"Error during search: {e}")
        return {"error": "An error occurred during the search."}

    results = []
    for filename in similar_pdfs:
        try:
            # Get or create the RAG chain for the specific PDF.
            rag_chain = get_or_create_pdf_rag_chain(app, filename, upload_folder)  # Pass 'app'


            if rag_chain is None: # Handling the None
                logging.error(f"Could not create or retrieve RAG chain for {filename}")
                results.append({
                    "filename": filename,
                    "error": "Could not initialize chat for this document."
                })
                continue


            download_link = url_for('download_pdf', filename=filename, _external=True)
            embed_link = url_for('view_pdf', filename=filename, _external=True)
            chat_link = url_for('pdf_specific_chat_page', filename=filename, _external=True)  # New chat link

            results.append({
                "filename": filename,
                "download_link": download_link,
                "embed_link": embed_link,
                "chat_link": chat_link,  # Add the chat link to the result

            })
        except Exception as e:
            logging.exception(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "error": "Could not process this document."
            })

    logging.info(f"Found similar PDFs: {results}")
    return {"results": results}