import argparse
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer, util

"""Load and preprocess movie dataset from CSV."""
def load_dataset(file_path):
    dataset = pd.read_csv(file_path, low_memory=False)
    dataset = dataset.dropna(subset=['overview'])
    dataset['description'] = dataset['title'].astype(str) + " " + dataset['overview'].astype(str)  # Combine title & overview
    
    # Compute IMDB-like weighted rating parameters
    C = dataset['vote_average'].mean()
    m = dataset['vote_count'].quantile(0.9)
    dataset['score'] = weighted_rating(dataset, m, C)
    # dataset = dataset.loc[dataset['vote_count'] >= m].copy()
    
    return dataset, m, C

"""Calculate IMDB-style weighted rating."""
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

"""Compute SBERT embeddings for movie overviews."""
def compute_sbert_embeddings(dataset, model):
    return model.encode(dataset['description'].tolist(), show_progress_bar=True, convert_to_tensor=True)

"""Recommend movies using TF-IDF and cosine similarity."""
def recommend_movies_tfidf(user_input, dataset, tfidf_vectorizer, tfidf_matrix, m, C, top_n=5):
    # Transform user input using the same vectorizer
    user_vector = tfidf_vectorizer.transform([user_input])  # Use transform, not fit_transform

    cosine_sim = linear_kernel(user_vector, tfidf_matrix)[0]  # Ensure dimensions match
    top_indices = cosine_sim.argsort()[-top_n:][::-1]

    recommendations = []
    for index in top_indices:
        row = dataset.iloc[index].copy()
        row['similarity'] = cosine_sim[index]
        # row['score'] = weighted_rating(row, m, C)
        recommendations.append(row)

    return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)

"""Recommend movies using SBERT embeddings and cosine similarity."""
def recommend_movies_sbert(user_input, dataset, movie_embeddings, model, m, C, top_n=10):
    user_embedding = model.encode(user_input, show_progress_bar=True, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, movie_embeddings)[0]
    top_indices = torch.argsort(similarities.cpu(), descending=True)[:top_n]
    
    recommendations = []
    for index in top_indices:
        row = dataset.iloc[index.item()].copy()
        row['similarity'] = similarities[index].item()
        # row['score'] = weighted_rating(row, m, C)
        recommendations.append(row)
    
    return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)

"""Display recommended movies."""
def display_recommendations(recommendations, method_name):
    print(f"\nTop {len(recommendations)} Recommended Movies using {method_name}:\n")
    for i, row in enumerate(recommendations, start=1):
        print(f"{i}. Title: \033[92m{row['title']}\033[0m\n   Similarity Score: \033[93m{(row['similarity']*100):.2f}%\033[0m\n") #   Overview: {row['overview']}\n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('user_input', type=str, help='User input for movie preferences')
    args = parser.parse_args()
    
    if not args.user_input.strip():
        args.user_input = input("Please enter your movie preferences: ")
    
    file_path = "./archive/movies_metadata.csv"
    dataset, m, C = load_dataset(file_path)
    
    # Compute feature vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['description'].astype(str).tolist()) #compute_tfidf_vectors(dataset)
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    movie_embeddings = compute_sbert_embeddings(dataset, sbert_model)
    
    # Generate recommendations
    tfidf_recommendations = recommend_movies_tfidf(args.user_input, dataset, tfidf_vectorizer, tfidf_matrix, m, C, 5)
    display_recommendations(tfidf_recommendations, "TF-IDF")
    
    sbert_recommendations = recommend_movies_sbert(args.user_input, dataset, movie_embeddings, sbert_model, m, C, 5)
    display_recommendations(sbert_recommendations, "SBERT")