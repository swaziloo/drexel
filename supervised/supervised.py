import pandas as pd
import random
import numpy as np
import os, glob, sys
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

# --- DATA HARVESTING ---
def load_and_preprocess_original_logic(tmdb_path, data_root):
    print("\nLoading Movie Data...")
    np.random.seed(707)
    
    tmdb = pd.read_csv(tmdb_path)
    tmdb['year'] = pd.to_datetime(tmdb['release_date'], errors='coerce').dt.year
    tmdb_truth = tmdb[['original_title', 'year', 'overview']].dropna()
    
    all_files = glob.glob(os.path.join(data_root, "*/imdb_movies_*.csv"))
    imdb_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df['Title'] = df['Title'].str.replace(r'^\d+\.\s+', '', regex=True)
        imdb_list.append(df)
    
    imdb_master = pd.concat(imdb_list, axis=0, ignore_index=True)
    
    # Matched pairs: Label 1
    positives = pd.merge(
        imdb_master[['Title', 'Year', 'description']], 
        tmdb_truth[['original_title', 'year', 'overview']], 
        left_on=['Title', 'Year'], 
        right_on=['original_title', 'year']
    )
    positives['label'] = 1

    # Mismatched title/years (assume remake): Label 0
    hard_negs = pd.merge(
        imdb_master[['Title', 'Year', 'description']], 
        tmdb_truth[['original_title', 'year', 'overview']], 
        left_on=['Title'], 
        right_on=['original_title']
    )
    hard_negs = hard_negs[hard_negs['Year'] != hard_negs['year']].sample(min(400, len(hard_negs)))
    hard_negs['label'] = 0

    # Random noise: Label 0
    size = min(500, len(imdb_master), len(tmdb_truth))
    random_indices_imdb = np.random.choice(imdb_master.index, size=size)
    random_indices_tmdb = np.random.choice(tmdb_truth.index, size=size)
    random_negs = pd.DataFrame({
        'Title': imdb_master.loc[random_indices_imdb, 'Title'].values,
        'description': imdb_master.loc[random_indices_imdb, 'description'].values,
        'Year': imdb_master.loc[random_indices_imdb, 'Year'].values,
        'original_title': tmdb_truth.loc[random_indices_tmdb, 'original_title'].values,
        'overview': tmdb_truth.loc[random_indices_tmdb, 'overview'].values,
        'year': tmdb_truth.loc[random_indices_tmdb, 'year'].values,
        'label': 0
    })

    full_df = pd.concat([positives, hard_negs, random_negs], ignore_index=True)

    # Print Dataset Statistics (Assess Class Balance)
    print("\n---- DATASET COMPOSITION STATISTICS ----")
    print(f"Total Movies Processed: {len(full_df)}")
    print(f"  [+] True Matches (Label 1):       {len(full_df[full_df['label'] == 1])}")
    print(f"  [-] Hard Negatives (Shuffled):    {len(full_df[(full_df['label'] == 0) & (full_df['Title'] == full_df['original_title'])])}")
    print(f"  [-] Random Negatives (Noise):     {len(full_df[(full_df['label'] == 0) & (full_df['Title'] != full_df['original_title'])])}")
    print("-" * 40)

    return full_df

# --- JITTER LOGIC ---
# Corrupts a specific percentage of 'Year' values
def apply_custom_jitter(df, percentage):
    df_j = df.copy()
    if percentage <= 0:
        return df_j
        
    np.random.seed(42 + percentage)
    mask = df_j.sample(frac=percentage/100, random_state=42 + percentage).index
    for idx in mask:
        # Assign a random outlier year
        outlier = np.random.choice([1900, 2026, 1970, 2025])
        df_j.loc[idx, 'Year'] = outlier
    
    return df_j

# --- EXPERIMENT ENGINE ---
def run_experiment():
    imdb_path = os.path.join('movie_data', 'Data')
    tmdb_path = os.path.join('movie_data', 'tmdb_5000_movies.csv')
    data = load_and_preprocess_original_logic(tmdb_path, imdb_path)
    
    active_transformer = None
    current_model_name = ""
    working_df = data.copy()
    current_jitter = 0

    while True:
        if active_transformer is None:
            print("\n=== SELECT TRANSFORMER ===")
            print("1: MiniLM | 2: MPNet | 3: Multi-QA | Q: Quit")
            t_choice = input("Choice: ").upper()
            if t_choice == 'Q': break
            
            models = {"1": "all-MiniLM-L6-v2", "2": "all-mpnet-base-v2", "3": "multi-qa-mpnet-base-dot-v1"}
            current_model_name = models.get(t_choice, "all-MiniLM-L6-v2")
            active_transformer = SentenceTransformer(current_model_name)
            
            print(f"Encoding text with {current_model_name}...")
            emb_tmdb = active_transformer.encode(data['overview'].tolist(), show_progress_bar=True)
            emb_imdb = active_transformer.encode(data['description'].tolist(), show_progress_bar=True)
            
            data['similarity_score'] = util.cos_sim(emb_tmdb, emb_imdb).diag().tolist()
            working_df = data.copy()

        print(f"\n=== SETTINGS (Model: {current_model_name} | Jitter: {current_jitter}%) ===")
        print("J: New Transformer | 1: LogReg | 2: GradBoost | %: Set Jitter % |  Q: Quit")
        action = input("Action: ").upper()

        if action == 'Q': break
        if action == 'J':
            current_jitter = 0
            active_transformer = None
            continue
            
        if action == '%':
            try:
                current_jitter = int(input("Enter Jitter Percentage (0-100): "))
                working_df = apply_custom_jitter(data, current_jitter)
                print(f"Jitter updated to {current_jitter}%")
            except ValueError:
                print("Invalid input. Please enter a number.")
            continue

        # Prep: calculate difference between TMDB year and IMDb Year
        working_df['year_diff'] = (working_df['year'] - working_df['Year']).abs()
        X = working_df[['similarity_score', 'year_diff']]
        y = working_df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clfs = {"1": ("LogisticRegression", LogisticRegression()), 
                "2": ("GradientBoosting", GradientBoostingClassifier())}
        
        clf_tuple = clfs.get(action)
        if clf_tuple is not None:
            name, clf = clf_tuple
            # Scaled values for human-readable weights
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Training and Results Block
            print(f"Training {name} with {current_jitter}% metadata jitter...")
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            # Text Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print("\n---- TEXT CONFUSION MATRIX ------------------")
            print(f"{'':<15} | Predicted NO | Predicted YES")
            print(f"{'-'*45}")
            print(f"{'Actual NO':<15} | {tn:<12} | {fp:<12}")
            print(f"{'Actual YES':<15} | {fn:<12} | {tp:<12}")
            print(f"{'-'*45}")

            # Supervised Learning Metrics
            print(f"\nTP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}\n")
            print(classification_report(y_test, y_pred, zero_division=0))

            if name == "GradientBoosting":
                importances = clf.feature_importances_
                print(f"Similarity Score Importance: {importances[0]:.2f}")
                print(f"Year Difference Importance: {importances[1]:.2f}")

            elif name == "LogisticRegression":
                coeffs = clf.coef_[0]
                total = np.abs(coeffs).sum()
                print(f"Similarity Score Weight: {np.abs(coeffs[0])/total:.2f}")
                print(f"Year Difference Weight: {np.abs(coeffs[1])/total:.2f}")
            
if __name__ == "__main__":
    run_experiment()
