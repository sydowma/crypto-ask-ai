import chromadb
from sentence_transformers import SentenceTransformer
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variables (especially for API keys)
load_dotenv()

# --- Configuration --- (Keep these easily accessible)
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "okx_content"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct' # Embedding model
N_RESULTS = 3 # Adjust number of results for UI

# --- LLM Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "qwq" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# --- Model and DB Loading (Cached) ---
@st.cache_resource
def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def connect_to_chromadb():
    print(f"Connecting to ChromaDB client at path: {VECTOR_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to collection: '{COLLECTION_NAME}'")
        return collection
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}. Run build_vector_db.py first.")
        return None

# --- API Call Functions --- (Remain largely the same)
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
        response = requests.post(api_url, json=payload, headers=headers, timeout=60) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Parse the JSON response line by line if streaming, or directly if not
        return response.json().get('response', 'Error: No response field in Ollama output.')
    except requests.exceptions.Timeout:
        return "Error: Ollama API request timed out."
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
            ],
            timeout=60 # Added timeout
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# --- Prompt Formatting --- (Remains the same)
def format_prompt(query, context_docs):
    """Formats the prompt for the LLM using retrieved context."""
    context = "\n\n---\n\n".join(context_docs)
    prompt = f"Based on the following context, please answer the query.\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    return prompt

# --- Core Query Logic ---
def perform_rag_query(query, llm_choice, ollama_model, openai_model):
    embed_model = load_embedding_model()
    collection = connect_to_chromadb()

    if not embed_model or not collection:
        return None, "Error: Could not load model or connect to DB.", []

    # Embed Query
    try:
        query_embedding = embed_model.encode(query).tolist()
    except Exception as e:
        st.error(f"Error embedding query: {e}")
        return None, "Error embedding query.", []

    # Query ChromaDB
    retrieved_docs_info = []
    context_docs = []
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances'] 
        )
        
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            context_docs = documents
            for i in range(len(ids)):
                 retrieved_docs_info.append({
                     "distance": distances[i],
                     "url": metadatas[i].get('url', 'N/A'),
                     "content_preview": documents[i][:300] + "..." # Short preview for UI
                 })
        else:
             st.warning("No relevant documents found in the database.")

    except Exception as e:
        st.error(f"Error querying ChromaDB: {e}")
        return None, "Error querying database.", []

    # Generate LLM response if selected
    llm_answer = "*LLM generation not selected.*"
    if llm_choice != 'None':
        if not context_docs:
             st.warning("No relevant context found. Asking LLM without retrieved context.")
        
        prompt = format_prompt(query, context_docs)
        
        with st.spinner(f"Querying {llm_choice}..."):
            if llm_choice == 'Ollama':
                llm_answer = get_ollama_response(prompt, ollama_model)
            elif llm_choice == 'OpenAI':
                llm_answer = get_openai_response(prompt, openai_model)
    
    return retrieved_docs_info, llm_answer

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 OKX Content Query (RAG)")

# Sidebar for configuration (optional, keep simple for now)
# st.sidebar.header("Configuration")
# N_RESULTS = st.sidebar.slider("Number of results to retrieve", 1, 10, N_RESULTS)

# LLM Selection
llm_option = st.radio(
    "Choose LLM for Answer Generation:",
    ('None', 'Ollama', 'OpenAI'), 
    horizontal=True,
    index=0 # Default to None
)

# Specific model selection (conditional)
ollama_model_name = DEFAULT_OLLAMA_MODEL
openai_model_name = DEFAULT_OPENAI_MODEL
if llm_option == 'Ollama':
    ollama_model_name = st.text_input("Ollama Model Name", DEFAULT_OLLAMA_MODEL)
if llm_option == 'OpenAI':
    openai_model_name = st.text_input("OpenAI Model Name", DEFAULT_OPENAI_MODEL)

# User Query Input
user_query = st.text_area("Enter your query about OKX content:", height=100)

# Submit Button
if st.button("Submit Query"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing query..."):
            # Run the RAG process
            retrieved_info, final_answer = perform_rag_query(
                user_query, 
                llm_option, 
                ollama_model_name, 
                openai_model_name
            )

        st.divider()
        
        # Display Final Answer
        st.subheader("🧠 Generated Answer")
        st.markdown(final_answer)
        
        st.divider()
        
        # Display Retrieved Documents (optional)
        if retrieved_info:
            st.subheader("📚 Retrieved Context Documents")
            for i, doc_info in enumerate(retrieved_info):
                with st.expander(f"Document {i+1} (Distance: {doc_info['distance']:.4f})"):
                    st.markdown(f"**URL:** [{doc_info['url']}]({doc_info['url']})")
                    st.caption(doc_info['content_preview'])
        elif llm_option == 'None':
             st.info("No documents retrieved and no LLM selected.")

# --- Removed Argparse Logic ---
# The __main__ block with argparse is no longer needed when running with Streamlit 