# download_models.py
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Set cache directory
cache_dir = "/app/models"
os.makedirs(cache_dir, exist_ok=True)

# Download models
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

for model_name in models:
    print(f"Downloading {model_name}...")
    SentenceTransformer(model_name, cache_folder=cache_dir)
    print(f"Downloaded {model_name}")