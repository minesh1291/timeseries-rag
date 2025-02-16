# Example Time Series Datasets

This directory contains example time series datasets for testing and demonstration purposes.
Each dataset is provided in CSV format with timestamp and value columns.

## Dataset Categories

### Synthetic Patterns (`synthetic/`)

Basic time series patterns for testing and validation:

- `sine_amplitude.csv`: Sine wave with increasing amplitude
- `composite.csv`: Composite signal with multiple frequencies
- `square_noisy.csv`: Square wave with added noise
- `trend_seasonal.csv`: Trend with seasonal component and noise

### Sensor Data (`sensor/`)

Simulated sensor readings with realistic patterns:

- `temperature.csv`: Temperature sensor with daily and weekly cycles
- `vibration.csv`: Vibration sensor with intermittent spikes
- `pressure.csv`: Pressure sensor with gradual drift

### Financial Data (`financial/`)

Simulated financial time series:

- `stock_price.csv`: Stock price with trend and volatility
- `volume.csv`: Trading volume with intraday patterns
- `volatility.csv`: Volatility clustering pattern

### Weather Data (`weather/`)

Simulated weather measurements:

- `temperature.csv`: Temperature with seasonal and daily patterns
- `humidity.csv`: Humidity with inverse correlation to temperature
- `wind_speed.csv`: Wind speed with gusts

## Data Format

All files follow the same format:
```csv
timestamp,value
2024-01-01 00:00:00,23.5
2024-01-01 01:00:00,24.2
...
```

## Usage Examples

### Python
```python
import pandas as pd

# Load a dataset
df = pd.read_csv('synthetic/sine_amplitude.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic statistics
print(df.describe())

# Plot the time series
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'])
plt.title('Sine Wave with Increasing Amplitude')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

### Time Series RAG
```python
from timeseries_rag.models import TimeSeriesEmbedder
from timeseries_rag.rag import TimeSeriesRAG, TimeSeriesDocument

# Load data
df = pd.read_csv('weather/temperature.csv')
time_series = df['value'].values

# Create embedder and RAG system
embedder = TimeSeriesEmbedder()
rag = TimeSeriesRAG()

# Add to database
embedding = embedder.embed(time_series)
doc = TimeSeriesDocument(
    id="weather_temp_1",
    data=time_series,
    metadata={"type": "weather", "measurement": "temperature"},
    embedding=embedding
)
rag.add_document(doc)
```

## Data Generation

The datasets were generated using Python scripts with controlled random seeds
for reproducibility. Each category simulates specific characteristics:

- Synthetic: Basic mathematical patterns with noise
- Sensor: Realistic sensor behavior with artifacts
- Financial: Market-like patterns with volatility
- Weather: Natural cycles and correlations

## Use Cases

1. Pattern Recognition
   - Find similar patterns in different time periods
   - Detect anomalies and outliers
   - Identify seasonal patterns

2. Time Series Classification
   - Distinguish between different types of patterns
   - Categorize sensor behaviors
   - Identify weather patterns

3. Similarity Search
   - Find similar market conditions
   - Identify similar weather patterns
   - Match sensor behavior patterns

4. Anomaly Detection
   - Detect unusual sensor readings
   - Identify market anomalies
   - Find weather anomalies