# download_model.py
from sentence_transformers import SentenceTransformer

# Pre-download the SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")