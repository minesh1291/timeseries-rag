"""Time Series Retrieval Augmented Generation (RAG) Module.

This module provides functionality for storing and retrieving time series data using
vector similarity search. It implements a RAG system specifically designed for time
series data, allowing efficient storage and retrieval of similar patterns.

Example:
    >>> from timeseries_rag.models import TimeSeriesEmbedder
    >>> embedder = TimeSeriesEmbedder()
    >>> rag = TimeSeriesRAG()
    >>> 
    >>> # Add a document
    >>> ts_data = np.sin(np.linspace(0, 10, 100))
    >>> embedding = embedder.embed(ts_data)
    >>> doc = TimeSeriesDocument(
    ...     id="sin_wave_1",
    ...     data=ts_data,
    ...     metadata={"type": "sine", "frequency": 1.0},
    ...     embedding=embedding
    ... )
    >>> rag.add_document(doc)
    >>> 
    >>> # Search for similar patterns
    >>> query = np.sin(np.linspace(0, 10, 100) + 0.1)
    >>> query_embedding = embedder.embed(query)
    >>> results = rag.search(query_embedding, k=5)
"""

import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

@dataclass
class TimeSeriesDocument:
    """A dataclass representing a time series document with metadata and embedding.
    
    This class stores all information related to a time series, including its raw
    data, metadata, and vector embedding for similarity search.
    
    Attributes:
        id (str): Unique identifier for the time series.
        data (np.ndarray): Raw time series data.
        metadata (Dict[str, Any]): Additional information about the time series.
        embedding (Optional[np.ndarray]): Vector embedding of the time series,
            used for similarity search. Default is None.
    
    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        >>> doc = TimeSeriesDocument(
        ...     id="example_1",
        ...     data=data,
        ...     metadata={"type": "example"},
        ...     embedding=np.array([0.1, 0.2, 0.3])
        ... )
    """
    
    id: str
    data: np.ndarray
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class TimeSeriesRAG:
    """A class implementing Retrieval Augmented Generation for time series data.
    
    This class provides functionality for storing time series documents and
    retrieving similar patterns using FAISS vector similarity search.
    
    Attributes:
        embedding_dim (int): Dimension of the time series embeddings.
        index (faiss.Index): FAISS index for similarity search.
        documents (List[TimeSeriesDocument]): List of stored time series documents.
    
    Example:
        >>> rag = TimeSeriesRAG(embedding_dim=260)
        >>> doc = TimeSeriesDocument(...)
        >>> rag.add_document(doc)
        >>> results = rag.search(query_embedding, k=5)
    """
    
    def __init__(self, embedding_dim: int = 260):
        """Initialize the TimeSeriesRAG system.
        
        Args:
            embedding_dim (int, optional): Dimension of the time series embeddings.
                Should match the output dimension of your embedding model.
                Defaults to 260 (256 resampled points + 4 statistical features).
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[TimeSeriesDocument] = []
        
    def add_document(self, doc: TimeSeriesDocument) -> None:
        """Add a time series document to the RAG system.
        
        Args:
            doc (TimeSeriesDocument): The document to add. Must have a valid
                embedding for similarity search.
        
        Raises:
            ValueError: If the document's embedding is None or has incorrect shape.
        """
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
            
        if doc.embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {doc.embedding.shape[-1]}"
            )
            
        self.index.add(doc.embedding.reshape(1, -1))
        self.documents.append(doc)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar time series patterns.
        
        Args:
            query_embedding (np.ndarray): The embedding vector of the query time
                series. Must match the embedding dimension of the index.
            k (int, optional): Number of nearest neighbors to retrieve.
                Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search results.
                Each dictionary has the following keys:
                - 'id': Document ID
                - 'distance': L2 distance to query
                - 'data': Raw time series data
                - 'metadata': Document metadata
        
        Raises:
            ValueError: If query_embedding has incorrect shape.
        """
        if query_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {query_embedding.shape[-1]}"
            )
            
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'id': doc.id,
                    'distance': float(distances[0][i]),
                    'data': doc.data.tolist(),
                    'metadata': doc.metadata
                })
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[TimeSeriesDocument]:
        """Retrieve a document by its ID.
        
        Args:
            doc_id (str): The ID of the document to retrieve.
        
        Returns:
            Optional[TimeSeriesDocument]: The document if found, None otherwise.
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None