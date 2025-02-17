# Time Series RAG

[![Documentation Status](https://github.com/minesh-1291/timeseries-rag/actions/workflows/docs.yml/badge.svg)](https://minesh-1291.github.io/timeseries-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful system for time series similarity search and retrieval augmented generation (RAG). This project provides an efficient way to search for similar time series patterns and augment them with contextual information.

## Project Structure

```
timeseries_rag/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       └── azure-deploy.yml    # Azure deployment configuration
├── data/                       # Example datasets
│   ├── financial/             # Financial time series data
│   ├── sensor/                # Sensor readings data
│   ├── synthetic/             # Synthetic patterns
│   └── weather/               # Weather measurements
├── docs/                       # Documentation
│   └── docs/
│       ├── source/            # Sphinx documentation source
│       └── requirements.txt    # Documentation dependencies
├── scripts/                    # Utility scripts
│   └── start.sh               # Azure startup script
├── src/                       # Source code
│   └── timeseries_rag/        # Main package
│       ├── __init__.py        # Package initialization
│       ├── api.py             # FastAPI web application
│       ├── models.py          # Time series embedding models
│       └── rag.py             # RAG system implementation
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # MIT License
├── MANIFEST.in                # Package manifest
├── README.md                  # This file
├── requirements.txt           # Project dependencies
└── setup.py                   # Package setup configuration
```

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

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/minesh-1291/timeseries-rag.git
cd timeseries-rag
```

2. Install in development mode:
```bash
pip install -e .
```

### Production Installation

```bash
pip install timeseries-rag
```

## Usage

### Running the Web Application

Development mode:
```bash
python -m timeseries_rag.api
```

Production mode:
```bash
gunicorn timeseries_rag.api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Access the web interface at:
- Development: http://localhost:8000
- Production: http://your-domain:8000

### Python API

```python
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument
import numpy as np

# Initialize components
embedder = TimeSeriesEmbedder(target_length=256)
rag = TimeSeriesRAG()

# Create some example time series
t = np.linspace(0, 10, 100)
sine_wave = np.sin(t)
noisy_sine = sine_wave + np.random.normal(0, 0.2, size=len(sine_wave))

# Embed and store the first time series
embedding = embedder.embed(sine_wave)
doc = TimeSeriesDocument(
    id="sine_1",
    data=sine_wave,
    metadata={"type": "sine", "frequency": 1.0},
    embedding=embedding
)
rag.add_document(doc)

# Search for similar patterns
query_embedding = embedder.embed(noisy_sine)
results = rag.search(query_embedding, k=5)

# Print results
for result in results:
    print(f"Document ID: {result['id']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Metadata: {result['metadata']}")
    print()
```

## Documentation

Full documentation is available at [https://minesh-1291.github.io/timeseries-rag/](https://minesh-1291.github.io/timeseries-rag/)

## Deployment

### Azure App Service

1. Configure GitHub Actions:
   - Add Azure publish profile as secret
   - Enable GitHub Actions workflow

2. Deploy using GitHub Actions:
   - Push to main branch
   - Monitor deployment in Actions tab

3. Access your application:
   - Web Interface: https://your-app.azurewebsites.net
   - API Documentation: https://your-app.azurewebsites.net/docs

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
  url = {https://github.com/minesh1291/timeseries-rag}
}
```
