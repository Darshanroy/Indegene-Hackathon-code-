# PDF Powerhouse: Integrate, Search, and Chat with Your PDFs

PDF Powerhouse is a Flask-based web application that allows you to integrate multiple PDF documents into a single, searchable knowledge base.  Perform semantic searches across all your PDFs, and engage in interactive chats with individual documents using a powerful Retrieval-Augmented Generation (RAG) pipeline.

## Features

*   **F1: PDF Integration and Indexing:**
    *   Seamlessly integrate multiple PDF files.
    *   Automatic indexing of PDF content using FAISS for rapid similarity search.
    *   Persistent index storage for efficient retrieval.

*   **F2: Global Semantic Search:**
    *   Perform keyword-based semantic searches across *all* integrated PDFs.
    *   Results display relevant PDF filenames, download links, and embedded previews.
    *   Utilizes Sentence Transformers for high-quality embeddings.

*   **F3: Per-Document Chat (RAG):**
    *   Chat with *individual* PDFs returned by the search.
    *   Employs a dedicated LangChain RAG pipeline for each PDF.
    *   Provides contextually relevant answers extracted directly from the specific PDF's content.
    *   Leverages Groq for fast and accurate responses.
    *   Maintains distinct chat histories for each PDF and for general inquiries.

*   **F4: General Chat (RAG):**
    *   Ask questions related to the entire collection of integrated PDFs.
    *   Maintains chat history.

*   **User-Friendly Interface:**
    *   Clean, intuitive web interface built with HTML, CSS, and JavaScript.
    *   Responsive design for optimal viewing across devices.

*   **Logging:**
    *   Comprehensive logs are written to the `logs/` directory.
    *   Utilizes the standard Python `logging` module at the `INFO` level, providing insights into application operation.

## Installation and Setup

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Highly Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` File:**

    Create a file named `.env` in the project's root directory.  This file stores API keys and configuration settings. Add the following, replacing placeholders with your actual values:

    ```
    COHERE_API_KEY=your_cohere_api_key  # Optional (remove if not using Cohere)
    GROQ_API_KEY=your_groq_api_key
    HF_TOKEN=your_huggingface_api_token
    ```

    *   **`COHERE_API_KEY`:** (Optional) Your API key for Cohere (if using Cohere embeddings).
    *   **`GROQ_API_KEY`:** Your API key for Groq (used for the LLM in the RAG pipeline).
    *   **`HUGGINGFACE_API_TOKEN`:** Your Hugging Face API token (for Sentence Transformers).
    
5.  **Run the Application:**

    ```bash
    python app.py
    ```

    The application will be accessible at `http://0.0.0.0:5000` in your web browser.

## Usage

1.  **Integrate PDFs:** On the main page, use the file integration form to select and add one or more PDF files to your knowledge base.

2.  **Search:** Navigate to the "Search PDFs" page and enter your search query. Results display matching PDFs with links for downloading, viewing, and chatting.

3.  **Chat with a Specific PDF:** After searching, click "Chat with PDF" next to a result to open a chat interface dedicated to that PDF.

4.  **Chat with All PDFs:** Go to the "Chat with PDFs" page to ask questions about the entire collection.

5.  **Restart:** Use the "Restart" button to clear the application and integrate new PDFs.


## Troubleshooting

*   **ChromaDB Errors:** Ensure `CHROMA_DB_PATH` is an *absolute* path in `.env` and the directory is writable. Delete the contents of `CHROMA_DB_PATH` to force a rebuild.
*   **Missing API Keys:** Verify that you have set `GROQ_API_KEY` and `HUGGINGFACE_API_TOKEN` in your `.env` file.
*   **RAG Issues:** Double-check that `CHROMA_DB_PATH` is identical in all files. Examine the logs for errors.
*   **Dependencies:** If you encounter problems, run `pip install -r requirements.txt` to ensure all libraries are installed.

## Contributing

Pull requests are welcome!

This improved `README.md` uses more precise language ("integrate" instead of "upload"), provides clearer instructions, and emphasizes the importance of absolute paths for reliable operation. It also includes a more detailed explanation of the configuration options in the `.env` file.

