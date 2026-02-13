## Amazon Music Recommender ##
Utilizes the [Stanford SNAP Amazon product co-purchasing network metadata](https://snap.stanford.edu/data/amazon-meta.html)
to find top-recommended albums based on customer ratings.

### Setup ###
Configure base environment:
* Install `python3` (tested with Python3 on Debian in WSL)
* Install `python3-dev` build tools

Configure Virtual Environment for testing (recommended):
* `python3 -m venv rec_env` creates a virtual env in the current folder
* `source rec_env/bin/activate` move your shell session to the virtual env

Install Dependencies:
* `pip install "numpy<2.0"` a recent update does not work with Surprise
* `pip install scikit-surprise` the main Surprise library
* `pip install pandas` to parse TSV data

### Fetch the Amazon Metadata and parse out Music ratings ###
* `curl -O -L https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz` fetch the data archive
* `gunzip amazon-meta.txt.gz` unpack the archive
* `python3 ParseAmazonFiltered.py amazon-meta.txt` extract ratings
* `python3 TestRecs.py` calculate the RMSE and F1@K using 20% of the data set for test

### Prepare Test Ratings ###

You can choose one of the Customer IDs from within the `customer_ratings_filtered.txt` file or add a `TESTUSER` customer to the file.
A separate TESTUSER.tsv is available with 25 ratings. If you choose a different Customer, you need to change that value in the `FindTopRecs.py` script.

### Getting Recommendations ###
* `python3 FindTopRecs.py` provides the highest 5 ratings from the dataset.