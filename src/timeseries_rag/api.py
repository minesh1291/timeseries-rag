"""FastAPI Web Application for Time Series RAG.

This module provides a web interface and REST API for the Time Series RAG system.
It allows users to upload time series data, add metadata, and search for similar
patterns through a user-friendly interface.

The module includes:
- REST API endpoints for uploading and searching time series
- Interactive web interface with Plotly visualizations
- CORS middleware for cross-origin requests
- File upload handling for CSV data
- Error handling and validation

Example:
    To run the web application:
    
    ```python
    import uvicorn
    from timeseries_rag.api import app
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=50758)
    ```
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Optional
import uuid

from .models import TimeSeriesEmbedder
from .rag import TimeSeriesRAG, TimeSeriesDocument

# Create FastAPI application
app = FastAPI(
    title="Time Series RAG",
    description="Time series similarity search and retrieval augmented generation",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embedder = TimeSeriesEmbedder()
rag_system = TimeSeriesRAG()

@app.post("/upload")
async def upload_timeseries(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
) -> Dict[str, str]:
    """Upload a time series file with optional metadata.
    
    This endpoint accepts a CSV file containing time series data and optional
    metadata in JSON format. The time series is embedded and stored in the RAG
    system for later retrieval.
    
    Args:
        file (UploadFile): CSV file containing time series data. Should have
            one or more columns of numerical values.
        metadata (Optional[str], optional): JSON string containing metadata
            about the time series. Defaults to None.
    
    Returns:
        Dict[str, str]: A dictionary containing:
            - status: "success" if upload was successful
            - document_id: UUID of the stored document
    
    Raises:
        HTTPException: If file reading, parsing, or storage fails.
    
    Example:
        ```python
        import requests
        
        files = {'file': open('timeseries.csv', 'rb')}
        metadata = '{"type": "temperature", "location": "sensor1"}'
        response = requests.post(
            'http://localhost:50758/upload',
            files=files,
            data={'metadata': metadata}
        )
        print(response.json())
        ```
    """
    try:
        content = await file.read()
        df = pd.read_csv(content)
        time_series = df.values
        
        # Generate embedding
        embedding = embedder.embed(time_series)
        
        # Create document
        doc_id = str(uuid.uuid4())
        metadata_dict = json.loads(metadata) if metadata else {}
        
        doc = TimeSeriesDocument(
            id=doc_id,
            data=time_series,
            metadata=metadata_dict,
            embedding=embedding
        )
        
        # Add to RAG system
        rag_system.add_document(doc)
        
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    k: int = 5
) -> Dict[str, List[Dict]]:
    """Search for similar time series patterns.
    
    This endpoint accepts a CSV file containing a query time series and returns
    the k most similar time series from the database.
    
    Args:
        file (UploadFile): CSV file containing the query time series data.
        k (int, optional): Number of similar patterns to retrieve. Defaults to 5.
    
    Returns:
        Dict[str, List[Dict]]: A dictionary containing:
            - results: List of similar time series, each with:
                - id: Document ID
                - distance: L2 distance to query
                - data: Time series values
                - metadata: Document metadata
    
    Raises:
        HTTPException: If file reading, parsing, or search fails.
    
    Example:
        ```python
        import requests
        
        files = {'file': open('query.csv', 'rb')}
        response = requests.post(
            'http://localhost:50758/search',
            files=files,
            params={'k': 10}
        )
        print(response.json())
        ```
    """
    try:
        content = await file.read()
        df = pd.read_csv(content)
        query_ts = df.values
        
        # Generate embedding
        query_embedding = embedder.embed(query_ts)
        
        # Search similar
        results = rag_system.search(query_embedding, k=k)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the main web interface.
    
    Returns:
        str: HTML content for the web interface.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Time Series RAG</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .section { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Time Series RAG System</h1>
            
            <div class="section">
                <h2>Upload Time Series</h2>
                <form id="uploadForm">
                    <input type="file" id="uploadFile" accept=".csv">
                    <input type="text" id="metadata" placeholder="Metadata (JSON)">
                    <button type="submit">Upload</button>
                </form>
            </div>
            
            <div class="section">
                <h2>Search Similar Time Series</h2>
                <form id="searchForm">
                    <input type="file" id="searchFile" accept=".csv">
                    <button type="submit">Search</button>
                </form>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('uploadFile').files[0]);
                formData.append('metadata', document.getElementById('metadata').value);
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                alert('Upload successful! Document ID: ' + result.document_id);
            };
            
            document.getElementById('searchForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('searchFile').files[0]);
                
                const response = await fetch('/search', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<h2>Search Results</h2>';
                
                result.results.forEach((item, index) => {
                    const div = document.createElement('div');
                    div.innerHTML = `<h3>Result ${index + 1}</h3>`;
                    
                    const plotDiv = document.createElement('div');
                    plotDiv.id = `plot${index}`;
                    div.appendChild(plotDiv);
                    
                    resultsDiv.appendChild(div);
                    
                    Plotly.newPlot(`plot${index}`, [{
                        y: item.data,
                        type: 'scatter'
                    }]);
                });
            };
        </script>
    </body>
    </html>
    """

def main():
    """Run the FastAPI application using uvicorn.
    
    This function is the entry point for running the web application. It configures
    uvicorn with the appropriate host and port settings.
    """
    uvicorn.run(
        "timeseries_rag.api:app",
        host="0.0.0.0",
        port=50758,
        reload=True
    )

if __name__ == "__main__":
    main()