import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TimeSeriesDocument:
    id: str
    data: np.ndarray
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class TimeSeriesRAG:
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[TimeSeriesDocument] = []
        
    def add_document(self, doc: TimeSeriesDocument):
        if doc.embedding is not None:
            self.index.add(doc.embedding.reshape(1, -1))
            self.documents.append(doc)
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
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
    
    def get_document_by_id(self, doc_id: str):
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None