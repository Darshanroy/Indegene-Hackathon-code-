from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Incorrect
# from langchain_huggingface import HuggingFaceEndpointEmbeddings  # Incorrect
from langchain_cohere import CohereEmbeddings  # Correct import
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token and Groq API key from environment variables
# HF_TOKEN = os.getenv("HF_TOKEN") #No longer needed
COHERE_API_KEY = os.getenv("COHERE_API_KEY") # Use Cohere API key now
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
app = Flask(__name__)

# Configuration
DATA_FOLDER = "data"
PDF_DIRECTORY = r"database\upload"  # Default PDF directory
CHUNK_SIZE = 700
CHROMA_DB_PATH = "database\indexing\chroma_langchain_db"
#EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" #No Longer Needed - Using cohere.


# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


# Instead of globals, use Flask's app context or a database/cache
def get_chat_session_history(session_id: str) -> BaseChatMessageHistory:
    """Gets or creates a chat history for a given session."""
    # Access the chat history store from the Flask app's config
    chat_history_store = app.config.setdefault('chat_history_store', {})

    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]



def load_documents_from_directory(directory_path):
    """Loads PDF documents from a directory using PyPDFDirectoryLoader."""
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return None

    loader = PyPDFDirectoryLoader(directory_path)
    try:
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents from directory: {e}")
        return None


def initialize_vector_store(documents):
    """Initializes and populates a Chroma vector store from a list of documents."""
    if not documents:
        print("Warning: No documents to process.  Ensure the directory contains PDF files.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Use CohereEmbeddings with API key from env variable
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY # Load cohere API Key
    )


    vector_store = Chroma(
        collection_name="Patient_data",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    try:
        vector_store.add_documents(documents=docs)
        return vector_store
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return None


def create_conversational_rag_chain(llm, vector_store):
    """Creates a conversational RAG chain for question answering."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )



    qa_prompt_template = ChatPromptTemplate.from_template("""
**Prompt:**
**Context:**
{context}
**Question:**
{input}
**Instructions:**
1. **Carefully read and understand the provided context.**
2. **Think step-by-step to formulate a comprehensive and accurate answer.**
3. **Base your response solely on the given context.**
4. **Ensure the answer is clear, concise, and easy to understand.**
5. **Ensure the answer is in small understandable points with all content.**
**Response:**
[Your detailed and well-reasoned answer]
**Note:** This prompt emphasizes careful consideration and accurate response based on the provided context.
""")

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)

    history_aware_retriever = create_history_aware_retriever(
        llm,
        vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 10, 'fetch_k': 50}
        ),
        contextualize_q_prompt
    )

    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_chat_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Endpoint to process PDF data and create the RAG chain."""
    # Store in the app context, not globals
    app.config['conversational_rag_chain'] = None
    app.config['vector_store'] = None


    data = request.get_json()
    pdf_directory = data.get('pdf_directory', PDF_DIRECTORY)  # Use the default if not provided

    documents = load_documents_from_directory(pdf_directory)
    if not documents:
        return jsonify({"error": "Failed to load documents."}), 400

    vector_store = initialize_vector_store(documents)
    if not vector_store:
        return jsonify({"error": "Failed to initialize vector store."}), 500

    # if not GROQ_API_KEY:
    #     return jsonify({"error": "GROQ_API_KEY not found in environment variables."}), 500

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

    conversational_rag_chain = create_conversational_rag_chain(llm, vector_store)

    # Store in the app config
    app.config['conversational_rag_chain'] = conversational_rag_chain
    app.config['vector_store'] = vector_store

    return jsonify({"message": "PDF data processed successfully."}), 200


@app.route('/query', methods=['POST'])
def query():
    """Endpoint to query the RAG chain."""

    data = request.get_json()
    user_question = data.get('question')
    session_id = data.get('session_id', 'default_session')  # Provide a default session ID if none given


    if not user_question:
        return jsonify({"error": "Missing question."}), 400

    # Access from app context
    conversational_rag_chain = app.config.get('conversational_rag_chain')
    vector_store = app.config.get('vector_store')


    if conversational_rag_chain is None or vector_store is None:
        # Reinitialize if needed, especially if the server restarts
        print("Reinitializing RAG chain...")

        # groq_api = os.environ.get("groq_api")
        # if not groq_api:
        #     return jsonify({"error": "GROQ_API_KEY not found in environment variables."}), 500

        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

        if not os.path.exists(CHROMA_DB_PATH): # Check if DB exists.  If not re-initialize.
            print("Chroma DB not found.  Processing PDF and re-creating.")
            documents = load_documents_from_directory(PDF_DIRECTORY) #  Load from default location.  Need a better way to persist this.
            if not documents:
                return jsonify({"error": "Failed to load documents for re-initialization."}), 500

            vector_store = initialize_vector_store(documents)
            if not vector_store:
                return jsonify({"error": "Failed to initialize vector store."}), 500

        else:
            #Re-Initialize Vector store and load from persistence.
            embeddings = CohereEmbeddings(
                model="embed-english-v3.0",
                cohere_api_key=COHERE_API_KEY # Load cohere API Key
                )

            vector_store = Chroma(
                collection_name="Patient_data",
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_PATH,
            )

        conversational_rag_chain = create_conversational_rag_chain(llm, vector_store)
        app.config['conversational_rag_chain'] = conversational_rag_chain
        app.config['vector_store'] = vector_store




    try:
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = response['answer']
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error generating answer: {e}")
        return jsonify({"error": f"Error generating answer: {e}"}), 500

@app.route('/')
def index():
    return render_template('index-2.html')

if __name__ == '__main__':
    # Use app context for initialization
    with app.app_context():
        app.config['conversational_rag_chain'] = None
        app.config['vector_store'] = None

    app.run(debug=True, port=5000)  # Or any other port you prefer