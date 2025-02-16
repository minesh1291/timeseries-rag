import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

class TimeSeriesEmbedder:
    def __init__(self, target_length=256):
        self.target_length = target_length
        self.scaler = StandardScaler()
    
    def embed(self, time_series):
        """Convert time series to embedding vector using resampling and statistical features"""
        if isinstance(time_series, list):
            time_series = np.array(time_series)
            
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
            
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