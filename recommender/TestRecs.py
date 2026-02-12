import pandas as pd
from surprise import SVDpp, Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    # Map predictions to each user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

# read reviews
df = pd.read_csv('customer_ratings_filtered.txt', sep='\t')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['CustomerID', 'ASIN', 'Rating']], reader)

# split set and train on 80%
trainset, testset = train_test_split(data, test_size=0.2)
algo = SVD(n_factors=20, reg_all=0.1)
algo.fit(trainset)

# predict on the held-out 20%
predictions = algo.test(testset)

# calculate RMSE
rmse = accuracy.rmse(predictions)

# determine the precision of the top-5
precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4.0)
avg_p = sum(p for p in precisions.values()) / len(precisions)
print(f"Precision@5: {avg_p:.4f}")

