"""Pattern Recognition Examples for Time Series RAG.

This script demonstrates advanced pattern recognition capabilities using
Time Series RAG and the analytics module.
"""

import numpy as np
import matplotlib.pyplot as plt
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument
from timeseries_rag.analytics import TimeSeriesAnalytics

def extract_patterns_example():
    """Example of extracting and analyzing patterns.
    
    This example demonstrates:
    1. Creating synthetic time series data
    2. Using TimeSeriesAnalytics to extract patterns
    3. Visualizing the extracted patterns
    """
    # Create synthetic data with multiple patterns
    t = np.linspace(0, 20, 1000)
    
    # Combine different patterns
    data = (
        np.sin(t) +                     # Base sine wave
        0.5 * np.sin(2 * t) +           # Higher frequency component
        0.2 * np.sin(0.5 * t) +         # Lower frequency component
        0.1 * np.random.randn(len(t))   # Random noise
    )

    # Initialize analytics
    analytics = TimeSeriesAnalytics(data)

    # Extract patterns
    patterns = analytics.extract_patterns(window_size=100, n_patterns=3)

    # Visualize patterns
    plt.figure(figsize=(15, 5))
    for i, pattern in enumerate(patterns):
        plt.subplot(1, 3, i+1)
        plt.plot(pattern['pattern'])
        plt.title(f'Pattern {i+1}\nFrequency: {pattern["frequency"]:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

    return patterns

def pattern_search_example(patterns):
    """Example of searching for similar patterns.
    
    Args:
        patterns: List of patterns extracted from extract_patterns_example()
    
    This example shows:
    1. Adding extracted patterns to RAG system
    2. Searching for similar patterns
    3. Visualizing search results
    """
    # Initialize RAG system
    embedder = TimeSeriesEmbedder()
    rag = TimeSeriesRAG()

    # Add patterns to database
    for i, pattern in enumerate(patterns):
        embedding = embedder.embed(pattern['pattern'])
        doc = TimeSeriesDocument(
            id=f'pattern_{i}',
            data=pattern['pattern'],
            metadata={'frequency': pattern['frequency']},
            embedding=embedding
        )
        rag.add_document(doc)

    # Create a query pattern (slightly modified version of first pattern)
    query = patterns[0]['pattern'] + np.random.normal(0, 0.1, 
                                                     size=len(patterns[0]['pattern']))
    query_embedding = embedder.embed(query)
    results = rag.search(query_embedding, k=2)

    # Visualize results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(query)
    plt.title('Query Pattern')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result['data'], 
                label=f'Pattern {result["id"]}\nDistance: {result["distance"]:.2f}')
    plt.title('Similar Patterns')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def seasonality_analysis_example():
    """Example of analyzing seasonality in time series.
    
    This example demonstrates:
    1. Creating time series with known seasonality
    2. Detecting seasonality using analytics
    3. Visualizing the results
    """
    # Create synthetic data with multiple seasonal components
    t = np.linspace(0, 30, 1000)
    
    # Daily and weekly patterns
    daily = np.sin(2 * np.pi * t / 1)   # 1-day period
    weekly = 0.5 * np.sin(2 * np.pi * t / 7)  # 7-day period
    
    # Combine patterns with noise
    data = daily + weekly + 0.1 * np.random.randn(len(t))

    # Analyze seasonality
    analytics = TimeSeriesAnalytics(data)
    seasonality = analytics.detect_seasonality(max_period=10)

    # Print results
    print("Seasonality Analysis:")
    print(f"Detected period: {seasonality['period']:.2f}")
    print(f"Strength: {seasonality['strength']:.2f}")

    # Visualize data and detected seasonality
    plt.figure(figsize=(12, 4))
    plt.plot(t[:100], data[:100], label='Data')
    plt.axvline(x=seasonality['period'], color='r', linestyle='--',
                label=f'Detected Period: {seasonality["period"]:.2f}')
    plt.title('Time Series with Detected Seasonality')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Running pattern extraction example...")
    patterns = extract_patterns_example()
    
    print("\nRunning pattern search example...")
    pattern_search_example(patterns)
    
    print("\nRunning seasonality analysis example...")
    seasonality_analysis_example()