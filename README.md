# Time Series RAG

[![Documentation Status](https://github.com/minesh-1291/timeseries-rag/actions/workflows/docs.yml/badge.svg)](https://minesh-1291.github.io/timeseries-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful system for time series similarity search and retrieval augmented generation (RAG). This project provides an efficient way to search for similar time series patterns and augment them with contextual information.

## Features

### Time Series Embedding
- Statistical feature extraction
- Resampling to fixed length
- Efficient dimensionality reduction
- Support for variable-length time series

### Similarity Search
- FAISS vector database integration
- Fast nearest neighbor search
- Customizable distance metrics
- Metadata-aware retrieval

### Web Interface
- Interactive time series visualization
- Easy data upload and search
- Real-time results display
- Plotly-powered charts

### API
- RESTful endpoints
- FastAPI backend
- CORS and iframe support
- Easy integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/minesh-1291/timeseries-rag.git
cd timeseries-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
python -m timeseries_rag.api
```

Access the web interface at http://localhost:50758

### Python API

```python
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG

# Initialize components
embedder = TimeSeriesEmbedder()
rag = TimeSeriesRAG()

# Add time series to the database
time_series = [1.0, 2.0, 3.0, 2.0, 1.0]
embedding = embedder.embed(time_series)
rag.add_document({
    'id': 'ts1',
    'data': time_series,
    'embedding': embedding,
    'metadata': {'type': 'example'}
})

# Search for similar patterns
query = [1.0, 2.0, 3.0, 2.5, 1.5]
query_embedding = embedder.embed(query)
results = rag.search(query_embedding, k=5)
```

## Documentation

Full documentation is available at [https://minesh-1291.github.io/timeseries-rag/](https://minesh-1291.github.io/timeseries-rag/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{timeseries_rag2024,
  author = {Jethva, Minesh A.},
  title = {Time Series RAG},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/minesh-1291/timeseries-rag}
}
```