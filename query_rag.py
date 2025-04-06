import chromadb
from sentence_transformers import SentenceTransformer
import argparse
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (especially for API keys)
load_dotenv()

# --- Configuration ---
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "okx_content"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct' # Embedding model
N_RESULTS = 5 # Number of relevant documents to retrieve

# --- LLM Configuration ---
# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default Ollama API URL
DEFAULT_OLLAMA_MODEL = "qwq" # Or your preferred Ollama model
# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini" # Or your preferred OpenAI model

def get_ollama_response(prompt, model_name):
    """Gets a response from the Ollama API."""
    api_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False # Keep it simple for now
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Parse the JSON response line by line if streaming, or directly if not
        return response.json().get('response', 'Error: No response field in Ollama output.')
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from Ollama."

def get_openai_response(prompt, model_name):
    """Gets a response from the OpenAI API."""
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY environment variable not set."
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on the provided context."}, 
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def format_prompt(query, context_docs):
    """Formats the prompt for the LLM using retrieved context."""
    context = "\n\n---\n\n".join(context_docs)
    prompt = f"Based on the following context, please answer the query.\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    return prompt

def main(query, llm_choice, ollama_model, openai_model):
    # 1. Initialize Embedding Model
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        return

    # 2. Connect to ChromaDB
    print(f"Connecting to ChromaDB client at path: {VECTOR_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to collection: '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error connecting to ChromaDB or getting collection: {e}")
        print("Please ensure you have run 'build_vector_db.py' first.")
        return

    # 3. Embed the Query
    print(f"Embedding query: '{query}'")
    try:
        query_embedding = embed_model.encode(query).tolist()
    except Exception as e:
        print(f"Error embedding query: {e}")
        return

    # 4. Query the Collection (Retrieve relevant documents)
    print(f"Querying vector database for {N_RESULTS} relevant documents...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances'] 
        )
    except Exception as e:
        print(f"Error querying ChromaDB collection: {e}")
        return

    print("\n--- Retrieved Documents ---")
    if not results or not results.get('ids') or not results['ids'][0]:
        print("No relevant documents found in the database.")
        # Decide if you still want to ask the LLM without context
        context_docs = [] 
    else:
        # Display retrieved docs (optional, can be verbose)
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        context_docs = documents # Use the retrieved documents as context

        for i in range(len(ids)):
            print(f"\nResult {i+1}:")
            # print(f"  ID:        {ids[i]}")
            print(f"  Distance:  {distances[i]:.4f}")
            print(f"  URL:       {metadatas[i].get('url', 'N/A')}")
            # print(f"  Title:     {metadatas[i].get('title', 'N/A')}")
            # print(f"  Content:   {documents[i][:200]}...") # Shorter preview
            print("---")

    # 5. Generate Answer using LLM (if an LLM is selected)
    llm_answer = "No LLM selected for generation."
    if llm_choice != 'none':
        if not context_docs:
             print("\nWarning: No relevant context found. Asking LLM without retrieved context.")
        
        # Format the prompt
        prompt = format_prompt(query, context_docs)
        print("\n--- Sending Prompt to LLM ---")
        # print(prompt) # Uncomment to see the full prompt sent to the LLM
        print("Querying LLM...")

        if llm_choice == 'ollama':
            llm_answer = get_ollama_response(prompt, ollama_model)
        elif llm_choice == 'openai':
            llm_answer = get_openai_response(prompt, openai_model)

    # 6. Display Final Answer
    print("\n============ FINAL ANSWER ============")
    print(llm_answer)
    print("=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the OKX vector database and optionally generate an answer using an LLM.")
    parser.add_argument("query", type=str, help="The query string to search for.")
    parser.add_argument("--llm", type=str, default='none', choices=['ollama', 'openai', 'none'], 
                        help="Choose the LLM for answer generation ('ollama', 'openai', or 'none' for retrieval only). Default: none")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL, 
                        help=f"Specify the Ollama model to use (if --llm=ollama). Default: {DEFAULT_OLLAMA_MODEL}")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL, 
                        help=f"Specify the OpenAI model to use (if --llm=openai). Default: {DEFAULT_OPENAI_MODEL}")
    
    args = parser.parse_args()
    
    main(args.query, args.llm, args.ollama_model, args.openai_model) 