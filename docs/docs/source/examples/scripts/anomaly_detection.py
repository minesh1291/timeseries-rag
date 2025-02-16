"""Anomaly Detection Examples for Time Series RAG.

This script demonstrates anomaly detection capabilities using Time Series RAG
and the analytics module.
"""

import numpy as np
import matplotlib.pyplot as plt
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument
from timeseries_rag.analytics import TimeSeriesAnalytics

def create_synthetic_sensor_data():
    """Create synthetic sensor data with anomalies.
    
    Returns:
        tuple: Three time series (temperature, vibration, pressure) with anomalies
    """
    # Time points
    t = np.linspace(0, 10, 1000)
    
    # Temperature with daily cycle and anomalies
    temp_base = 20 + 5 * np.sin(2 * np.pi * t / 1)  # Daily pattern
    temp_anomalies = np.zeros_like(t)
    temp_anomalies[300:320] = 10  # Sudden spike
    temp_anomalies[600:700] = 5   # Sustained deviation
    temperature = temp_base + temp_anomalies + np.random.normal(0, 0.5, size=len(t))
    
    # Vibration with intermittent spikes
    vibration = np.random.normal(0, 0.1, size=len(t))
    spike_locations = np.random.choice(len(t), size=10, replace=False)
    vibration[spike_locations] += np.random.uniform(1, 2, size=len(spike_locations))
    
    # Pressure with gradual drift and sudden drop
    pressure = 100 + 0.1 * t + np.random.normal(0, 0.1, size=len(t))
    pressure[800:] -= 5  # Sudden drop
    
    return temperature, vibration, pressure, t

def detect_anomalies_example():
    """Example of detecting anomalies in sensor data.
    
    This example demonstrates:
    1. Creating synthetic sensor data with known anomalies
    2. Using TimeSeriesAnalytics to detect anomalies
    3. Visualizing the results
    """
    # Get synthetic data
    temperature, vibration, pressure, t = create_synthetic_sensor_data()
    
    # Initialize analytics for each sensor
    temp_analytics = TimeSeriesAnalytics(temperature)
    vib_analytics = TimeSeriesAnalytics(vibration)
    press_analytics = TimeSeriesAnalytics(pressure)
    
    # Detect anomalies
    temp_anomalies = temp_analytics.detect_anomalies(window_size=50)
    vib_anomalies = vib_analytics.detect_anomalies(window_size=20)
    press_anomalies = press_analytics.detect_anomalies(window_size=100)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Temperature
    plt.subplot(3, 1, 1)
    plt.plot(t, temperature, label='Temperature')
    if temp_anomalies:
        anomaly_idx = [x[0] for x in temp_anomalies]
        plt.scatter(t[anomaly_idx], temperature[anomaly_idx],
                   color='red', label='Anomalies')
    plt.title('Temperature Sensor')
    plt.legend()
    
    # Vibration
    plt.subplot(3, 1, 2)
    plt.plot(t, vibration, label='Vibration')
    if vib_anomalies:
        anomaly_idx = [x[0] for x in vib_anomalies]
        plt.scatter(t[anomaly_idx], vibration[anomaly_idx],
                   color='red', label='Anomalies')
    plt.title('Vibration Sensor')
    plt.legend()
    
    # Pressure
    plt.subplot(3, 1, 3)
    plt.plot(t, pressure, label='Pressure')
    if press_anomalies:
        anomaly_idx = [x[0] for x in press_anomalies]
        plt.scatter(t[anomaly_idx], pressure[anomaly_idx],
                   color='red', label='Anomalies')
    plt.title('Pressure Sensor')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return temp_anomalies, vib_anomalies, press_anomalies

def analyze_anomaly_patterns(anomalies, data, window_size=50):
    """Extract and analyze patterns around anomalies.
    
    Args:
        anomalies: List of (index, value) tuples indicating anomalies
        data: Original time series data
        window_size: Size of window around anomalies to analyze
    
    Returns:
        List of windows around anomalies
    """
    windows = []
    for idx, _ in anomalies:
        start = max(0, idx - window_size//2)
        end = min(len(data), idx + window_size//2)
        windows.append(data[start:end])
    return windows

def anomaly_pattern_analysis_example():
    """Example of analyzing patterns in anomalies.
    
    This example demonstrates:
    1. Detecting anomalies in multiple sensors
    2. Extracting patterns around anomalies
    3. Using RAG to find similar anomaly patterns
    """
    # Get data and detect anomalies
    temperature, vibration, pressure, t = create_synthetic_sensor_data()
    temp_anomalies, vib_anomalies, press_anomalies = detect_anomalies_example()
    
    # Initialize RAG system
    embedder = TimeSeriesEmbedder()
    rag = TimeSeriesRAG()
    
    # Extract and store anomaly patterns
    for sensor_name, anomalies, data in [
        ('temperature', temp_anomalies, temperature),
        ('vibration', vib_anomalies, vibration),
        ('pressure', press_anomalies, pressure)
    ]:
        windows = analyze_anomaly_patterns(anomalies, data)
        for i, window in enumerate(windows):
            embedding = embedder.embed(window)
            doc = TimeSeriesDocument(
                id=f'{sensor_name}_anomaly_{i}',
                data=window,
                metadata={'sensor': sensor_name, 'type': 'anomaly'},
                embedding=embedding
            )
            rag.add_document(doc)
    
    # Find similar anomaly patterns
    if temp_anomalies:  # Use first temperature anomaly as query
        query_window = analyze_anomaly_patterns(
            [temp_anomalies[0]], temperature
        )[0]
        query_embedding = embedder.embed(query_window)
        results = rag.search(query_embedding, k=3)
        
        # Visualize similar anomalies
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(query_window)
        plt.title('Query Anomaly Pattern')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        plt.subplot(1, 2, 2)
        for result in results:
            plt.plot(result['data'],
                    label=f"{result['metadata']['sensor']}\n"
                          f"Distance: {result['distance']:.2f}")
        plt.title('Similar Anomaly Patterns')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Running anomaly detection example...")
    detect_anomalies_example()
    
    print("\nRunning anomaly pattern analysis example...")
    anomaly_pattern_analysis_example()