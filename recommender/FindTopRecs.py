import pandas as pd
from surprise import SVDpp, Dataset, Reader, SVD
df = pd.read_csv('customer_ratings_filtered.txt', sep='\t')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['CustomerID', 'ASIN', 'Rating']], reader)
trainset = data.build_full_trainset()
# 20 factors, stricter model
algo = SVDpp(n_factors=20, reg_all=0.1)
algo.fit(trainset)

def get_top_5(user_id):
    all_asins = df['ASIN'].unique()
    rated_asins = df[df['CustomerID'] == user_id]['ASIN'].tolist()
    unrated = [a for a in all_asins if a not in rated_asins]
    predictions = [algo.predict(user_id, asin) for asin in unrated]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:5]

# the user in the file for which to find ratings
top_5 = get_top_5('TESTUSER')
for p in top_5:
    print(f"ASIN: {p.iid} | Predicted Score: {p.est:.2f}")
