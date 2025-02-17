#!/bin/bash

# Install the package
pip install -e .

# Start the server
cd /home/site/wwwroot && gunicorn timeseries_rag.api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000