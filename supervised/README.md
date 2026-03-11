## Movie Match Supervised Learnig ##

This demo was designed and tested in a WSL environment with python3 installed.

### Setup ###
Download and unpack two data sets into the 'movie_data' subfolder:
https://www.kaggle.com/datasets/raedaddala/imdb-movies-from-1960-to-2023
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data

### Running the Demo ###

* Ensure the `runDemo.sh` script is executable:
  * `chmod 755 runDemo.sh`
* Execute by running it:
  * `./runDemo.sh`

The first time its run, the script will create a virtual env for the demo and download a list of dependencies.
*This may take several minutes.*

### Demo Commands ###
* `curl -O -L https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz` fetch the data archive
* `gunzip amazon-meta.txt.gz` unpack the archive
* `python3 ParseAmazonFiltered.py amazon-meta.txt` extract ratings
* `python3 TestRecs.py` calculate the RMSE and F1@K using 20% of the data set for test

### Prepare Test Ratings ###

You can choose one of the Customer IDs from within the `customer_ratings_filtered.txt` file or add a `TESTUSER` customer to the file.
A separate TESTUSER.tsv is available with 25 ratings. If you choose a different Customer, you need to change that value in the `FindTopRecs.py` script.

### Getting Recommendations ###
* `python3 FindTopRecs.py` provides the highest 5 ratings from the dataset.