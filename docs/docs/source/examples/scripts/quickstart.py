"""Quick Start Guide for Time Series RAG.

This script demonstrates basic usage of the Time Series RAG system.
"""

import numpy as np
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument

def basic_example():
    """Basic example of using Time Series RAG.
    
    This example shows how to:
    1. Create sample time series data
    2. Initialize the RAG system
    3. Add documents to the system
    4. Search for similar patterns
    """
    # Create sample data
    t = np.linspace(0, 10, 100)
    sine_wave = np.sin(t)
    noisy_sine = sine_wave + np.random.normal(0, 0.1, size=len(sine_wave))

    # Initialize components
    embedder = TimeSeriesEmbedder()
    rag = TimeSeriesRAG()

    # Add original sine wave to RAG system
    embedding = embedder.embed(sine_wave)
    doc = TimeSeriesDocument(
        id="sine_1",
        data=sine_wave,
        metadata={"type": "sine", "frequency": 1.0},
        embedding=embedding
    )
    rag.add_document(doc)

    # Search using noisy sine wave
    query_embedding = embedder.embed(noisy_sine)
    results = rag.search(query_embedding, k=5)

    # Print results
    for result in results:
        print(f"Document ID: {result['id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print()

def visualization_example():
    """Example showing how to visualize results.
    
    This example demonstrates:
    1. Creating multiple time series
    2. Adding them to the RAG system
    3. Visualizing search results
    """
    import matplotlib.pyplot as plt

    # Create sample data
    t = np.linspace(0, 10, 100)
    patterns = {
        'sine': np.sin(t),
        'cosine': np.cos(t),
        'square': np.sign(np.sin(t)),
        'trend': 0.1 * t + np.sin(t)
    }

    # Initialize components
    embedder = TimeSeriesEmbedder()
    rag = TimeSeriesRAG()

    # Add patterns to RAG system
    for name, data in patterns.items():
        embedding = embedder.embed(data)
        doc = TimeSeriesDocument(
            id=name,
            data=data,
            metadata={"type": name},
            embedding=embedding
        )
        rag.add_document(doc)

    # Create a query pattern (noisy sine)
    query = patterns['sine'] + np.random.normal(0, 0.1, size=len(t))
    query_embedding = embedder.embed(query)
    results = rag.search(query_embedding, k=3)

    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot query
    plt.subplot(1, 2, 1)
    plt.plot(t, query)
    plt.title('Query Pattern')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Plot results
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(t, result['data'], 
                label=f"{result['id']} (dist: {result['distance']:.2f})")
    plt.title('Similar Patterns')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running basic example:")
    basic_example()
    
    print("\nRunning visualization example:")
    visualization_example()