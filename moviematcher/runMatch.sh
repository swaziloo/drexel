#!/bin/bash
# Force the locale to UTF-8 to prevent the \x00 characters
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

# 1. Check for Python and Pip
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

cd ~/git/drexel/moviematcher || exit

# 2. Use temporary Virtual Environment (SLTest)
VENV_NAME="SLTest"
source $VENV_NAME/bin/activate > /dev/null 2>&1

# 3. Start the MCP Server
python3 -u movie_mcp_server.py 2> ~/mcp.debug.log

# Cleanup on exit
deactivate
