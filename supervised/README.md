## Movie Match Supervised Learning ##

This demo was designed and tested in a WSL environment on Windows 11 with python3 installed.

### Setup ###
Download and unpack the two Kaggle data sets into the `movie_data` subfolder:

https://www.kaggle.com/datasets/raedaddala/imdb-movies-from-1960-to-2023

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data

### Running the Demo ###

* Ensure the `runDemo.sh` script is executable:
  * `chmod 755 runDemo.sh`
* Execute it:
  * `./runDemo.sh`

The first time its run, the script will create a virtual env for the demo and download a list of dependencies.
*This may take several minutes.*

Once it's complete,the supervised demo will start and load the data.

### Demo Commands ###
```
=== SELECT TRANSFORMER ===
1: MiniLM | 2: MPNet | 3: Multi-QA | Q: Quit
```
* Select the desired sentence transformer for the test by choosing 1, 2, or 3

Once the encodings are complete you'll get to the main demo choices:

```
=== SETTINGS (Model: all-MiniLM-L6-v2 | Jitter: 0%) ===
J: New Transformer | 1: LogReg | 2: GradBoost | %: Set Jitter % |  Q: Quit
Action:
```
* `J` to return to the transformer selection menu
* `1` to train and display the results for the Logistic Regression classifier
* `2` to train and display the results for the Gradient Boosting classifier
* `%` followed by an integer to intentionally misalign a percentage of the 'Year' columns