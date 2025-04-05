import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

# Configuration
JSON_FILE_PATH = "site_map_links.json"  # Your input JSON file from the crawler
VECTOR_DB_PATH = "./chroma_db"  # Directory to store the vector database
COLLECTION_NAME = "okx_content"
MODEL_NAME = 'all-mpnet-base-v2' # A good starting point for sentence embeddings

# Function to generate a consistent ID from URL
def generate_doc_id(url):
    return hashlib.sha256(url.encode()).hexdigest()

# Define the specific boilerplate string to remove
# (Ensure this exactly matches what you see, including potential variations in spacing or truncation)
BOILERPLATE_TEXT = "跳转至主要内容 买币 快捷买币 流程简单，快速成交 C2C 买币 灵活选择，0 交易费 发现 市场 查看最新行情和交易大数据 机会 发掘最热、最新币种，及时捕捉市场机会 交易 交易类型 闪兑 币币兑换，0 费率，无滑点 现货 轻松买卖数字货币 合约 交易永续和交割合约，灵活使用杠杆 期权 利用市场波动，赚取收益，降低交易风险 盘前交易 在币种未上线阶段，提前交易其交割合约 交易工具 策略交易 多种智能策略，助您轻松交易 策略广场 创建策略 价差速递 为合约价差提供深度流动性 询价单 支持自定义多腿策略和大宗交易 金融 赚币 持币生币， 赚取收益 简单赚币 链上赚币 结构化产品 借贷 质押数字资产，满足您的投资和消费需求 Jumpstart 抢先发现全球优质新项目 公链 X Layer 探索 X Layer 进入 Web3 的世界 X Layer 点燃创意、引领创新 主网浏览器 主网链上数据 测试网浏览器 测试网链上数据 X Layer 生态 探索 X Layer DApp 开发者"
# Consider using regex or multiple patterns if the boilerplate varies slightly

def main():
    # 1. Load data from JSON file
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        print(f"Loaded {len(documents_data)} documents from {JSON_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {JSON_FILE_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {JSON_FILE_PATH}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        return

    # Filter out entries with no content or title (or handle them as needed)
    valid_documents = [
        doc for doc in documents_data 
        if doc.get('url') and (doc.get('title') or doc.get('content'))
    ]
    print(f"Found {len(valid_documents)} documents with URL and title/content.")
    if not valid_documents:
        print("No valid documents to process.")
        return

    # 2. Initialize Embedding Model
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        return

    # 3. Initialize ChromaDB
    print(f"Initializing ChromaDB client at path: {VECTOR_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        # Get or create the collection
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Optionally specify the embedding function if you want Chroma to handle it
            # metadata={"hnsw:space": "cosine"} # Default space is l2, cosine often works well for text
        )
        print(f"Using collection: '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error initializing ChromaDB or collection: {e}")
        return

    # 4. Process and Embed Documents
    print("Processing and embedding documents...")
    batch_size = 100 # Process in batches for efficiency
    all_ids = []
    all_embeddings = []
    all_metadatas = []
    all_docs_text = []

    for i in tqdm(range(0, len(valid_documents), batch_size), desc="Embedding batches"):
        batch_data = valid_documents[i:i+batch_size]
        
        ids = [generate_doc_id(doc['url']) for doc in batch_data]
        
        # Clean content and prepare texts for embedding
        texts_to_embed = []
        cleaned_docs_for_storage = []
        for doc in batch_data:
            raw_content = doc.get('content', '')
            # Remove the specific boilerplate text
            cleaned_content = raw_content.replace(BOILERPLATE_TEXT, '').strip()
            # Optional: Add more cleaning steps here (e.g., remove excess whitespace)
            # cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            
            # Format for embedding (Title + Cleaned Content)
            text_for_embedding = f"Title: {doc.get('title', '')}\nContent: {cleaned_content}".strip()
            texts_to_embed.append(text_for_embedding)
            cleaned_docs_for_storage.append(text_for_embedding) # Store the cleaned version

        metadatas = [{'url': doc['url'], 'title': doc.get('title', '')} for doc in batch_data]
        
        # Generate embeddings
        try:
            embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            continue # Skip this batch on error

        all_ids.extend(ids)
        all_embeddings.extend(embeddings)
        all_metadatas.extend(metadatas)
        all_docs_text.extend(cleaned_docs_for_storage) # Use the cleaned text for storage

    # 5. Add to ChromaDB Collection
    if all_ids:
        print(f"Adding {len(all_ids)} embeddings to ChromaDB collection...")
        try:
            collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                documents=all_docs_text # Store the text alongside the embedding
            )
            print("Embeddings added successfully.")
        except Exception as e:
            print(f"Error adding embeddings to ChromaDB: {e}")
    else:
        print("No embeddings were generated to add to the database.")

    print("Vector database build process complete.")

if __name__ == "__main__":
    main() 