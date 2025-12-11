"""
Vector Store for Crypto Intelligence System
FAISS-based vector storage for semantic search and similarity matching
"""
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Schema for vector search result"""
    text: str
    score: float
    metadata: Dict[str, Any]
    index: int


class VectorStore:
    """
    FAISS-based vector store for semantic search
    
    Features:
    - Sentence transformer embeddings
    - Fast similarity search with FAISS
    - Metadata storage and retrieval
    - Persistence to disk
    """
    
    def __init__(
        self,
        model_name: str = None,
        dimension: int = 384
    ):
        """
        Initialize vector store
        
        Args:
            model_name: Sentence transformer model name
            dimension: Embedding dimension
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self.dimension = dimension
        
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[Any] = None
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        
        self._initialized = False
        
        logger.info("vector_store_initialized")
    
    def _ensure_initialized(self) -> bool:
        """Ensure the vector store is initialized"""
        if self._initialized:
            return True
        
        if not FAISS_AVAILABLE:
            logger.warning("faiss_not_available")
            return False
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence_transformers_not_available")
            return False
        
        try:
            # Load embedding model
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product
            
            self._initialized = True
            logger.info(
                "vector_store_ready",
                model=self.model_name,
                dimension=self.dimension
            )
            return True
            
        except Exception as e:
            logger.error(f"vector_store_init_error: {e}")
            return False
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self._ensure_initialized():
            return None
        
        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"embed_error: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        if not self._ensure_initialized():
            return None
        
        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"embed_batch_error: {e}")
            return None
    
    def add(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> int:
        """
        Add texts to the vector store
        
        Args:
            texts: Texts to add
            metadata: Optional metadata for each text
            
        Returns:
            Number of texts added
        """
        if not self._ensure_initialized():
            return 0
        
        if not texts:
            return 0
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        if embeddings is None:
            return 0
        
        # Add to FAISS index
        self._index.add(embeddings)
        
        # Store texts and metadata
        self._texts.extend(texts)
        
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])
        
        logger.info(f"added_to_vector_store", count=len(texts))
        return len(texts)
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar texts
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if not self._ensure_initialized():
            return []
        
        if not self._texts:
            return []
        
        # Embed query
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, len(self._texts))
        scores, indices = self._index.search(query_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            
            results.append(SearchResult(
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadata[idx],
                index=int(idx)
            ))
        
        return results
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self._ensure_initialized():
            return 0.0
        
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity (embeddings are normalized)
        return float(np.dot(emb1, emb2))
    
    def remove(self, indices: List[int]) -> int:
        """
        Remove items by index
        Note: FAISS doesn't support deletion, so we rebuild the index
        
        Args:
            indices: Indices to remove
            
        Returns:
            Number of items removed
        """
        if not self._texts:
            return 0
        
        indices_set = set(indices)
        
        # Filter out removed items
        new_texts = []
        new_metadata = []
        
        for i, (text, meta) in enumerate(zip(self._texts, self._metadata)):
            if i not in indices_set:
                new_texts.append(text)
                new_metadata.append(meta)
        
        removed_count = len(self._texts) - len(new_texts)
        
        # Rebuild index
        self._texts = new_texts
        self._metadata = new_metadata
        
        if new_texts and self._ensure_initialized():
            embeddings = self.embed_batch(new_texts)
            if embeddings is not None:
                self._index = faiss.IndexFlatIP(self.dimension)
                self._index.add(embeddings)
        
        logger.info(f"removed_from_vector_store", count=removed_count)
        return removed_count
    
    def clear(self) -> None:
        """Clear all data from vector store"""
        if self._ensure_initialized():
            self._index = faiss.IndexFlatIP(self.dimension)
        
        self._texts = []
        self._metadata = []
        
        logger.info("vector_store_cleared")
    
    def save(self, path: str) -> bool:
        """
        Save vector store to disk
        
        Args:
            path: Directory path to save to
            
        Returns:
            True if saved successfully
        """
        if not self._ensure_initialized():
            return False
        
        try:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self._index, str(save_dir / "index.faiss"))
            
            # Save texts and metadata
            with open(save_dir / "data.pkl", "wb") as f:
                pickle.dump({
                    "texts": self._texts,
                    "metadata": self._metadata,
                    "dimension": self.dimension,
                    "model_name": self.model_name
                }, f)
            
            logger.info(f"vector_store_saved", path=path)
            return True
            
        except Exception as e:
            logger.error(f"vector_store_save_error: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load vector store from disk
        
        Args:
            path: Directory path to load from
            
        Returns:
            True if loaded successfully
        """
        try:
            load_dir = Path(path)
            
            if not load_dir.exists():
                logger.warning(f"vector_store_path_not_found: {path}")
                return False
            
            # Load data
            with open(load_dir / "data.pkl", "rb") as f:
                data = pickle.load(f)
            
            self._texts = data["texts"]
            self._metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.model_name = data["model_name"]
            
            # Ensure model is loaded
            if not self._ensure_initialized():
                return False
            
            # Load FAISS index
            self._index = faiss.read_index(str(load_dir / "index.faiss"))
            
            logger.info(
                f"vector_store_loaded",
                path=path,
                count=len(self._texts)
            )
            return True
            
        except Exception as e:
            logger.error(f"vector_store_load_error: {e}")
            return False
    
    @property
    def size(self) -> int:
        """Get number of items in store"""
        return len(self._texts)
    
    def get_text(self, index: int) -> Optional[str]:
        """Get text by index"""
        if 0 <= index < len(self._texts):
            return self._texts[index]
        return None
    
    def get_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata by index"""
        if 0 <= index < len(self._metadata):
            return self._metadata[index]
        return None


# Global vector store instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get global vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
