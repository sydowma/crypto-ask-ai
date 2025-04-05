# Crypto Exchange Website RAG Query System

This project implements a Retrieval-Augmented Generation (RAG) system that allows querying content scraped from the OKX website (specifically `www.okx.com`) using natural language.

It crawls specified sitemaps, processes and cleans the text content, indexes it into a local vector database (ChromaDB), and provides a command-line interface to query the data. The query script can retrieve relevant document snippets and optionally use an LLM (Ollama or OpenAI) to generate a synthesized answer based on the retrieved context.

## Features

*   **Web Crawling:** Uses Scrapy to crawl `okx.com` sitemaps (`default-index.xml`, `learn-index.xml`, etc.).
*   **Content Extraction:** Extracts page URL, title, and body text content.
*   **Filtering:** Focuses on English (`/en/`) and Simplified Chinese (`/zh-hans/`) pages, excluding others via `deny_paths`.
*   **Indexing:** Processes scraped data, cleans common boilerplate text, and generates vector embeddings using `sentence-transformers` (`all-mpnet-base-v2`).
*   **Vector Storage:** Stores embeddings and associated metadata/text in a persistent ChromaDB database locally.
*   **RAG Querying:** Embeds user queries and retrieves relevant documents from the vector store based on semantic similarity.
*   **LLM Integration:** Optionally generates answers using:
    *   Ollama (local LLMs like `gemma3:27b`)
    *   OpenAI API (models like `gpt-4o-mini`)
*   **Retrieval-Only Mode:** Can be used to just view the most relevant document snippets without LLM generation.

## Project Structure

```
├── link_crawler/             # Scrapy project for crawling
│   ├── link_crawler/
│   │   ├── spiders/
│   │   │   └── sitemap_spider.py # The OKX sitemap spider
│   │   └── ... (other Scrapy files)
│   └── scrapy.cfg
├── build_vector_db.py      # Script to create/update the vector database
├── query_rag.py            # Script to query the database (with optional LLM generation)
├── requirements.txt        # Python dependencies
├── site_map_links.json     # Default output file for the crawler (contains scraped data)
├── chroma_db/              # Directory where ChromaDB stores the vector index (auto-generated)
├── .env                    # Optional: For API keys (e.g., OPENAI_API_KEY)
└── README.md               # This file
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    This project uses `uv` for faster package management.
    ```bash
    # Install uv if you don't have it (e.g., pip install uv)
    uv pip install -r requirements.txt
    ```

4.  **Ollama Setup (Optional):**
    *   Install Ollama from [ollama.com](https://ollama.com/).
    *   Ensure the Ollama server is running.
    *   Pull the desired model(s): `ollama pull gemma3:27b` (or other models you wish to use).

5.  **OpenAI API Key Setup (Optional):**
    *   Create a file named `.env` in the project root.
    *   Add your OpenAI API key:
        ```.env
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   Ensure this file is added to your `.gitignore` if using version control.

## Usage

The process involves three main steps:

**Step 1: Crawl Website Content**

*   Navigate to the crawler directory:
    ```bash
    cd link_crawler
    ```
*   Run the Scrapy spider. The `-o ../site_map_links.json` flag saves the output JSON file to the project's root directory.
    ```bash
    # Ensure the output path is correct relative to the link_crawler dir
    scrapy crawl sitemap_links_spider -o ../site_map_links.json 
    ```
*   This command will crawl the specified sitemaps and create/overwrite `site_map_links.json` in the root directory.
*   Return to the root directory:
    ```bash
    cd ..
    ```

**Step 2: Build/Update the Vector Database**

*   Run the indexing script. This will process `site_map_links.json`, clean the content, generate embeddings, and store them in the `./chroma_db` directory.
    ```bash
    python build_vector_db.py
    ```
*   Re-run this script whenever you update the `site_map_links.json` file with new crawl data.

**Step 3: Query the Data**

*   Use the `query_rag.py` script to ask questions.

*   **Retrieval Only (No LLM Generation):**
    ```bash
    python query_rag.py "What are the benefits of OKTC?"
    # Or explicitly:
    python query_rag.py "What are the benefits of OKTC?" --llm none
    ```

*   **Query with Ollama:**
    ```bash
    # Use default Ollama model (gemma3:27b)
    python query_rag.py "What are the benefits of OKTC?" --llm ollama
    
    # Specify a different Ollama model
    python query_rag.py "What are the benefits of OKTC?" --llm ollama --ollama-model llama3 
    ```

*   **Query with OpenAI:**
    ```bash
    # Use default OpenAI model (gpt-4o-mini)
    python query_rag.py "What are the benefits of OKTC?" --llm openai
    
    # Specify a different OpenAI model
    python query_rag.py "What are the benefits of OKTC?" --llm openai --openai-model gpt-4-turbo
    ```

## Configuration

Key configuration variables can be found at the top of the scripts:

*   **`build_vector_db.py`:**
    *   `JSON_FILE_PATH`: Path to the crawler's output file.
    *   `VECTOR_DB_PATH`: Directory for ChromaDB storage.
    *   `COLLECTION_NAME`: Name of the ChromaDB collection.
    *   `MODEL_NAME`: Sentence transformer model for embeddings.
    *   `BOILERPLATE_TEXT`: Specific string to remove from content during cleaning.
*   **`query_rag.py`:**
    *   `VECTOR_DB_PATH`, `COLLECTION_NAME`: Must match the build script.
    *   `EMBEDDING_MODEL_NAME`: Embedding model (should match the build script).
    *   `N_RESULTS`: Number of documents to retrieve for context.
    *   `OLLAMA_BASE_URL`, `DEFAULT_OLLAMA_MODEL`: Settings for Ollama.
    *   `DEFAULT_OPENAI_MODEL`: Default OpenAI model.
    *   `OPENAI_API_KEY`: Loaded from the `.env` file.

## Potential Improvements

*   **Content Extraction:** Enhance the Scrapy spider with more specific CSS/XPath selectors to isolate main content and better exclude headers/footers/menus, reducing the need for post-crawl cleaning.
*   **Text Chunking:** For very long documents, split content into smaller, overlapping chunks before embedding to potentially improve retrieval relevance for specific queries.
*   **Error Handling:** Add more robust error handling throughout the scripts.
*   **Model Selection:** Experiment with different sentence transformer embedding models.
*   **UI:** Build a simple web interface (e.g., using Streamlit or Flask) for easier interaction.