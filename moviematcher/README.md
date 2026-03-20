## TMDB Movie Matcher MCP Service ##

This demo was designed and tested using Claude Desktop with a WSL environment on Windows 11 with python3 installed.

### Setup ###
Download and unpack the two Kaggle data sets into the `movie_data` subfolder:

https://www.kaggle.com/datasets/raedaddala/imdb-movies-from-1960-to-2023

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data

* Note that these can be copied over from the 'supervised' demo if already downloaded

### Running the Demo ###

* Ensure the `trainMatch.sh` and `runMatch.sh` scriptq are executable:
    * `chmod 755 *.sh`
* Train the matcher:
    * `./trainMatch.sh`

The first time its run, the script will create a virtual env for the demo and download a list of dependencies.
*This may take several minutes.*

After completion, the embeddings, clean movie data, and joblib files will be present.

### Add movie_matcher to Claude Desktop ###
Edit `claude_desktop_config.json`.
On my installation, the file is located in:
`c:\Users\userName\AppData\Roaming\Claude`

Add the following lines to the top-level list (under 'preferences' in mine)
```
  ,
  "mcpServers": {
    "movie_matcher": {
      "command": "wsl.exe",
      "args": [
        "-d", "debian", 
        "bash", "~/git/drexel/moviematcher/runMatch.sh"
      ]
    }
  }
```
* Note the leading comma--you're adding this json map to the existing document.
* Correct the path if you haven't checked out the code into this location

Restart Claude Desktop:
* Use File->Exit as just closing the window may not restart the desktop app
