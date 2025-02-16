"""Time Series Analytics Module.

This module provides advanced analytics capabilities for time series data,
including pattern recognition, anomaly detection, and classification.

Example:
    >>> from timeseries_rag.analytics import TimeSeriesAnalytics
    >>> import pandas as pd
    >>> 
    >>> # Load data
    >>> df = pd.read_csv('data/sensor/temperature.csv')
    >>> analytics = TimeSeriesAnalytics(df['value'].values)
    >>> 
    >>> # Detect anomalies
    >>> anomalies = analytics.detect_anomalies()
    >>> print(f"Found {len(anomalies)} anomalies")
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class TimeSeriesAnalytics:
    """Advanced analytics for time series data.
    
    This class provides methods for analyzing time series data, including
    pattern recognition, anomaly detection, and feature extraction.
    
    Attributes:
        data (np.ndarray): The input time series data.
        scaler (StandardScaler): Scaler for normalizing the data.
    
    Example:
        >>> analytics = TimeSeriesAnalytics(time_series)
        >>> seasonality = analytics.detect_seasonality()
        >>> print(f"Detected seasonality: {seasonality}")
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize the analytics module.
        
        Args:
            data (np.ndarray): Input time series data.
        """
        self.data = data
        self.scaler = StandardScaler()
        self._normalized_data = self.scaler.fit_transform(
            data.reshape(-1, 1)
        ).ravel()
    
    def detect_anomalies(
        self,
        window_size: int = 24,
        threshold: float = 3.0
    ) -> List[Tuple[int, float]]:
        """Detect anomalies using rolling statistics.
        
        This method uses a combination of Z-score and DBSCAN to detect
        anomalies in the time series.
        
        Args:
            window_size (int): Size of the rolling window. Defaults to 24.
            threshold (float): Z-score threshold for anomaly detection.
                Defaults to 3.0.
        
        Returns:
            List[Tuple[int, float]]: List of (index, value) pairs indicating
                anomalies in the time series.
        """
        # Rolling statistics
        rolling_mean = np.convolve(
            self._normalized_data,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        
        # Pad the rolling mean
        pad_size = len(self._normalized_data) - len(rolling_mean)
        rolling_mean = np.pad(
            rolling_mean,
            (pad_size, 0),
            mode='edge'
        )
        
        # Calculate z-scores
        z_scores = np.abs(self._normalized_data - rolling_mean)
        
        # Find anomalies
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        return [(idx, self.data[idx]) for idx in anomaly_indices]
    
    def detect_seasonality(
        self,
        max_period: int = 168  # One week in hours
    ) -> Dict[str, Union[int, float]]:
        """Detect seasonality in the time series.
        
        This method uses autocorrelation to detect periodic patterns in
        the time series.
        
        Args:
            max_period (int): Maximum period to consider. Defaults to 168
                (one week in hours).
        
        Returns:
            Dict[str, Union[int, float]]: Dictionary containing:
                - period: Detected seasonality period
                - strength: Strength of the seasonality (correlation)
        """
        # Calculate autocorrelation
        acf = np.correlate(self._normalized_data, self._normalized_data, mode='full')
        acf = acf[len(acf)//2:]
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf[:max_period])
        
        if len(peaks) == 0:
            return {'period': 0, 'strength': 0.0}
        
        # Find strongest peak
        peak_strengths = acf[peaks]
        strongest_peak = peaks[np.argmax(peak_strengths[1:])]
        
        return {
            'period': int(strongest_peak),
            'strength': float(acf[strongest_peak])
        }
    
    def extract_patterns(
        self,
        window_size: int = 24,
        n_patterns: int = 5
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """Extract representative patterns from the time series.
        
        This method uses sliding windows and clustering to find common
        patterns in the time series.
        
        Args:
            window_size (int): Size of the sliding window. Defaults to 24.
            n_patterns (int): Number of patterns to extract. Defaults to 5.
        
        Returns:
            List[Dict[str, Union[np.ndarray, float]]]: List of patterns,
                each containing:
                - pattern: The pattern values
                - frequency: How often the pattern appears
        """
        # Create sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(
            self._normalized_data,
            window_size
        )
        
        # Reduce dimensionality
        pca = PCA(n_components=min(window_size, 10))
        windows_pca = pca.fit_transform(windows)
        
        # Cluster patterns
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(windows_pca)
        
        # Extract representative patterns
        patterns = []
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise
        
        for label in unique_labels[:n_patterns]:
            pattern_indices = np.where(labels == label)[0]
            pattern = np.mean(windows[pattern_indices], axis=0)
            frequency = len(pattern_indices) / len(windows)
            
            patterns.append({
                'pattern': pattern,
                'frequency': frequency
            })
        
        return patterns
    
    def calculate_features(self) -> Dict[str, float]:
        """Calculate statistical features of the time series.
        
        Returns:
            Dict[str, float]: Dictionary of features including:
                - mean: Mean value
                - std: Standard deviation
                - skewness: Skewness
                - kurtosis: Kurtosis
                - trend: Linear trend coefficient
                - seasonality_strength: Strength of seasonality
                - entropy: Sample entropy
        """
        # Basic statistics
        features = {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'skewness': float(stats.skew(self.data)),
            'kurtosis': float(stats.kurtosis(self.data))
        }
        
        # Trend
        t = np.arange(len(self.data))
        trend_coef = np.polyfit(t, self.data, 1)[0]
        features['trend'] = float(trend_coef)
        
        # Seasonality
        seasonality = self.detect_seasonality()
        features['seasonality_strength'] = seasonality['strength']
        
        # Entropy (using bins for continuous data)
        hist, _ = np.histogram(self.data, bins='auto')
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features['entropy'] = float(entropy)
        
        return features