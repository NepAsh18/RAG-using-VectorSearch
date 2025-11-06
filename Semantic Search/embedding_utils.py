import os
import requests
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "sample_mflix")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "movies")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Hugging Face Embedding API URL
EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a given text using Hugging Face API."""
    response = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Embedding request failed: {response.status_code} - {response.text}")

    return response.json()[0] if isinstance(response.json()[0], list) else response.json()

def embed_movie_plots(limit: int = 100):
    """Generate and store embeddings for movie plots up to the given limit."""
    for doc in collection.find({"plot": {"$exists": True}}).limit(limit):
        if "plot_embedding_hf" not in doc:
            try:
                embedding = generate_embedding(doc["plot"])
                doc["plot_embedding_hf"] = embedding
                collection.replace_one({"_id": doc["_id"]}, doc)
                print(f"✅ Embedded: {doc['title']}")
            except Exception as e:
                print(f"❌ Failed for {doc['title']}: {e}")

def semantic_search(query: str, limit: int = 5):
    """Perform a semantic vector search in MongoDB."""
    query_vector = generate_embedding(query)

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "plot_embedding_hf",
                "numCandidates": 100,
                "limit": limit,
                "index": "PlotSemanticSearch"
            }
        }
    ])
    return list(results)
