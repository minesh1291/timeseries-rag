"""Time Series Embedding Module.

This module provides functionality for converting time series data into fixed-length
embeddings using a combination of resampling and statistical features. The embeddings
can be used for similarity search and retrieval tasks.

Example:
    >>> embedder = TimeSeriesEmbedder(target_length=256)
    >>> time_series = [1.0, 2.0, 3.0, 2.0, 1.0]
    >>> embedding = embedder.embed(time_series)
    >>> print(embedding.shape)
    (1, 260)  # 256 resampled points + 4 statistical features
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from typing import Union, List, Tuple

class TimeSeriesEmbedder:
    """A class for converting time series data into fixed-length embeddings.
    
    This class provides functionality to transform variable-length time series into
    fixed-length embeddings by combining resampled values with statistical features.
    The embeddings can be used for similarity search and other downstream tasks.
    
    Attributes:
        target_length (int): The desired length of the resampled time series.
            Default is 256 points.
        scaler (StandardScaler): A scikit-learn StandardScaler instance for
            normalizing the time series data.
    
    Example:
        >>> embedder = TimeSeriesEmbedder(target_length=128)
        >>> time_series = np.sin(np.linspace(0, 10, 1000))
        >>> embedding = embedder.embed(time_series)
        >>> print(f"Embedding shape: {embedding.shape}")
    """
    
    def __init__(self, target_length: int = 256):
        """Initialize the TimeSeriesEmbedder.
        
        Args:
            target_length (int, optional): The desired length of the resampled
                time series. Defaults to 256.
        """
        self.target_length = target_length
        self.scaler = StandardScaler()
    
    def embed(self, time_series: Union[List[float], np.ndarray]) -> np.ndarray:
        """Convert time series to embedding vector using resampling and statistical features.
        
        This method performs the following steps:
        1. Converts input to numpy array if necessary
        2. Reshapes to 2D array if necessary
        3. Normalizes the time series using StandardScaler
        4. Resamples to fixed length using scipy.signal.resample
        5. Extracts statistical features (mean, std, max, min)
        6. Combines resampled values with statistical features
        
        Args:
            time_series (Union[List[float], np.ndarray]): Input time series data.
                Can be a 1D list/array or 2D array with shape (n_samples, n_features).
        
        Returns:
            np.ndarray: A 2D array of shape (1, target_length + 4) containing the
                embedding vector. The first target_length elements are the resampled
                values, followed by mean, std, max, and min statistics.
        
        Raises:
            ValueError: If the input time series is empty or has invalid dimensions.
        """
        if isinstance(time_series, list):
            time_series = np.array(time_series)
            
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
            
        if time_series.size == 0:
            raise ValueError("Input time series is empty")
            
        # Normalize
        time_series = self.scaler.fit_transform(time_series)
        
        # Resample to fixed length
        resampled = resample(time_series, self.target_length)
        
        # Extract statistical features
        mean = np.mean(time_series, axis=0)
        std = np.std(time_series, axis=0)
        max_val = np.max(time_series, axis=0)
        min_val = np.min(time_series, axis=0)
        
        # Combine features
        features = np.concatenate([
            resampled.flatten(),
            mean,
            std,
            max_val,
            min_val
        ])
        
        return features.reshape(1, -1)