import pandas as pd
import numpy as np
import joblib
import os, glob, sys
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("Loading Ground Truth Data (TMDB)...")
tmdb_path = os.path.join('movie_data', 'tmdb_5000_movies.csv')
tmdb = pd.read_csv(tmdb_path)
tmdb['year'] = pd.to_datetime(tmdb['release_date'], errors='coerce').dt.year

# create the clean reference set
tmdb_truth = tmdb.dropna(subset=['original_title', 'year', 'overview']).reset_index(drop=True)

# generate resources for the MCP Server
print(f"Encoding {len(tmdb_truth)} reference movies for the MCP database...")
transformer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
db_embeddings = transformer.encode(tmdb_truth['overview'].tolist(), show_progress_bar=True)

np.save("tmdb_embeddings.npy", db_embeddings)
tmdb_truth.to_csv("tmdb_clean.csv", index=False)
print("Saved tmdb_embeddings.npy and tmdb_clean.csv")

# load IMDb training data
print("\nHarvesting IMDb data for training samples...")
data_root = os.path.join('movie_data', 'Data')
all_files = glob.glob(os.path.join(data_root, "*/imdb_movies_*.csv"))
imdb_list = []
for f in all_files:
    df = pd.read_csv(f)
    df['Title'] = df['Title'].str.replace(r'^\d+\.\s+', '', regex=True)
    imdb_list.append(df)
imdb_master = pd.concat(imdb_list, axis=0, ignore_index=True)

# positives (Label 1)
positives = pd.merge(
    imdb_master[['Title', 'Year', 'description']],
    tmdb_truth[['original_title', 'year', 'overview']],
    left_on=['Title', 'Year'],
    right_on=['original_title', 'year']
).copy()
positives['label'] = 1

# hard negatives (Label 0) - Same title, different years
hard_negs = pd.merge(
    imdb_master[['Title', 'Year', 'description']],
    tmdb_truth[['original_title', 'year', 'overview']],
    left_on=['Title'],
    right_on=['original_title']
)
hard_negs = hard_negs[hard_negs['Year'] != hard_negs['year']].sample(min(400, len(hard_negs))).copy()
hard_negs['label'] = 0

# random noise/mismatched movies (Label 0)
size = min(500, len(imdb_master), len(tmdb_truth))
random_negs = pd.DataFrame({
    'description': imdb_master.sample(size)['description'].values,
    'Year': imdb_master.sample(size)['Year'].values,
    'overview': tmdb_truth.sample(size)['overview'].values,
    'year': tmdb_truth.sample(size)['year'].values,
    'label': 0
})

train_df = pd.concat([positives, hard_negs, random_negs], ignore_index=True)

# jiter the year to teach the model that it's not 100% reliable
print("Applying year jitter and outliers to training data...")
np.random.seed(220)
# neighborhood jitter: +/- 2 years for 20% of samples
mask_nb = train_df.sample(frac=0.20, random_state=220).index
train_df.loc[mask_nb, 'year'] += np.random.choice([-2, -1, 1, 2], size=len(mask_nb))
# outlier jitter: year unset/default/current (wrong) for 5% of samples
mask_out = train_df.sample(frac=0.05, random_state=220).index
train_df.loc[mask_out, 'Year'] = np.random.choice([1900, 2026, 1970, 2025], size=len(mask_out))

# encode and extract features
print("Calculating similarity scores for training..."
emb_train_tmdb = transformer.encode(train_df['overview'].tolist(), show_progress_bar=True)
emb_train_imdb = transformer.encode(train_df['description'].tolist(), show_progress_bar=True)
train_df['similarity_score'] = util.cos_sim(emb_train_tmdb, emb_train_imdb).diag().tolist()
train_df['year_diff'] = (train_df['year'] - train_df['Year']).abs()

# train model
X = train_df[['similarity_score', 'year_diff']]
y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# save
joblib.dump(clf, "movie_model.joblib")
joblib.dump(scaler, "scaler.joblib")

importances = clf.feature_importances_
print(f"\nTraining Complete.")
print(f"Feature Importance -> Similarity: {importances[0]:.4f} | Year: {importances[1]:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, clf.predict(X_test_scaled)))