#!/bin/bash

# 1. Check for Python and Pip
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 2. Set up temporary Virtual Environment (SLTest)
VENV_NAME="SLTest"
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME..."
    python3 -m venv $VENV_NAME
fi
 
# 3. Activate and ensure Pip is installed locally
echo "Activating environment and installing dependencies..."
source $VENV_NAME/bin/activate

# This bootstraps pip into the venv if it's missing
python3 -m ensurepip --upgrade

# Now you can safely use it
python3 -m pip install --upgrade pip
python3 -m pip install pandas numpy matplotlib sentence-transformers scikit-learn

# 4. Check for Data Files
# Verifies the existence of the TMDB file and the IMDb directory structure
if [ ! -f "movie_data/tmdb_5000_movies.csv" ]; then
    echo "Error: tmdb_5000_movies.csv not found in current directory."
    deactivate
    exit 1
fi

if [ ! -d "movie_data/Data" ]; then
    echo "Error: Directory 'Data' not found."
    deactivate
    exit 1
fi

# 5. Execute the Script
echo "Starting Demo: supervised-5.py..."
python3 supervised.py

# Cleanup on exit
deactivate
