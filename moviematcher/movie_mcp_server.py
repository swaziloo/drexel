import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import torch
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer, util

mcp = FastMCP("TMDbMovieMatcher")

MODEL_PATH = "movie_model.joblib"
SCALER_PATH = "scaler.joblib"
TMDB_CSV = "tmdb_clean.csv"
EMBEDDINGS_PATH = "tmdb_embeddings.npy"
TMDB_DATA_PATH = "movie_data/tmdb_5000_movies.csv"

model = None
scaler = None
transformer = None
tmdb_data = None
tmdb_embeddings = None

def load_resources():
    """Load pre-trained artifacts once on server startup."""
    global model, scaler, transformer, tmdb_data, tmdb_embeddings

    print("Initializing Movie Matcher Resources...", file=sys.stderr)

    # Load the same transformer used in train.py
    transformer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Load the TMDb reference database
    tmdb_data = pd.read_csv(TMDB_CSV)
    # Load the saved .npy and .joblib files
    tmdb_embeddings = np.load(EMBEDDINGS_PATH)

    if len(tmdb_data) != len(tmdb_embeddings):
        print(f"COUNT MISMATCH: Data ({len(tmdb_data)}) and Embeddings ({len(tmdb_embeddings)}) count mismatch!", file=sys.stderr)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Resources Loaded Successfully.", file=sys.stderr)

@mcp.tool()
def match_movie(description: str, year: int) -> str:
    """
    Analyzes a movie description and year to find a match in the TMDB catalog.
    Returns the title and a confidence score.
    """
    query_emb = transformer.encode(description, convert_to_tensor=True)

    db_tensor = torch.from_numpy(tmdb_embeddings).cpu()
    cos_scores = util.cos_sim(query_emb, tmdb_embeddings)[0]
    best_idx = int(cos_scores.argmax())
    best_match = tmdb_data.iloc[best_idx]

    similarity_score = float(cos_scores[best_idx])

    try:
        db_year = int(best_match['year'])
        user_year = int(year)
        year_diff = abs(db_year - user_year)
    except (ValueError, TypeError, KeyError):
        year_diff = 10 # Penalty for missing year data

    features = np.array([[similarity_score, year_diff]])
    features_scaled = scaler.transform(features)

    # Predict probability of 'Match' (Class 1)
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(probabilities[1])

    response_data = {
        "match_status": "HIGH_CONFIDENCE" if confidence > 0.5 else "LOW_CONFIDENCE",
        "match_details": {
            "title": best_match['original_title'],
            "db_year": db_year,
            "similarity_score": round(similarity_score, 4),
            "confidence_pct": round(confidence * 100, 2)
        }
    }

    return json.dumps(response_data)

if __name__ == "__main__":
    load_resources()
    mcp.run()