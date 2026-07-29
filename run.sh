#!/bin/bash

# Start the FastAPI backend in the background
# We bind to 0.0.0.0 so it's accessible inside the container
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the API to start
sleep 5

# Start the Streamlit frontend in the foreground
# We bind to 0.0.0.0 and port 7860 (required by HF Spaces)
streamlit run src/ui.py --server.port 7860 --server.address 0.0.0.0
