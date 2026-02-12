import pandas as pd
from surprise import SVDpp, Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
df = pd.read_csv('customer_ratings_filtered.txt', sep='\t')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['CustomerID', 'ASIN', 'Rating']], reader)
# hold back 20% to use for a test set
trainset, testset = train_test_split(data, test_size=0.2)
# 20 factors, stricter model
algo = SVDpp(n_factors=20, reg_all=0.1)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
