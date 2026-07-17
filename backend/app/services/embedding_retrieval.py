import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ExampleRetriever:
    """Retrieves few-shot examples using semantic search over column names."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExampleRetriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        bank_path = Path(__file__).resolve().parent.parent / "models" / "few_shot_examples.json"
        
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                self.examples = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load few-shot examples from {bank_path}: {e}")
            self.examples = []
            
        if self.examples:
            # Precompute embeddings for all example column names
            column_names = [ex["column_name"] for ex in self.examples]
            self.embeddings = self.model.encode(column_names)
        else:
            self.embeddings = np.array([])
            
        self._initialized = True
        logger.info(f"Initialized ExampleRetriever with {len(self.examples)} examples.")

    def get_similar_examples(self, column_name: str, k: int = 3) -> list[dict[str, Any]]:
        """Find the top-k most similar examples to the given column_name."""
        if not self.examples or len(self.embeddings) == 0:
            return []
            
        query_embedding = self.model.encode([column_name])
        
        # Compute cosine similarities between query and all bank embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            ex = self.examples[idx].copy()
            ex["similarity"] = round(float(similarities[idx]), 4)
            results.append(ex)
            
        return results

# Singleton accessor
_retriever = None

def get_similar_examples(column_name: str, k: int = 3) -> list[dict[str, Any]]:
    global _retriever
    if _retriever is None:
        _retriever = ExampleRetriever()
    return _retriever.get_similar_examples(column_name, k)
