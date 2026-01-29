"""
Model Cache Service

In-memory cache for storing trained ML models and their metadata.
Used to enable live predictions in demo pages without retraining.
"""

import uuid
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading


class ModelCache:
    """Thread-safe in-memory cache for ML models."""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 1):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache (LRU eviction)
            ttl_hours: Time-to-live for cached models in hours
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.lock = threading.Lock()
    
    def store_model(
        self,
        model: Any,
        preprocessor: Any,
        scenario_id: str,
        scenario_data: Dict[str, Any],
        feature_info: Dict[str, Any],
        task_type: str,
        model_name: str
    ) -> str:
        """
        Store a trained model with all necessary metadata.
        
        Args:
            model: Trained sklearn/xgboost model
            preprocessor: Fitted preprocessor (ColumnTransformer)
            scenario_id: Scenario identifier (e.g., 'crypto_signals')
            scenario_data: Scenario metadata (name, icon, description, etc.)
            feature_info: Information about input features (names, types, ranges)
            task_type: Type of task ('classification', 'regression', etc.)
            model_name: Name of the model (e.g., 'Random Forest Classifier')
        
        Returns:
            session_id: Unique identifier for retrieving the model
        """
        with self.lock:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Clean up expired entries
            self._cleanup_expired()
            
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Store model data
            self.cache[session_id] = {
                'model': model,
                'preprocessor': preprocessor,
                'scenario_id': scenario_id,
                'scenario_data': scenario_data,
                'feature_info': feature_info,
                'task_type': task_type,
                'model_name': model_name,
                'created_at': time.time(),
                'last_accessed': time.time()
            }
            
            return session_id
    
    def get_model(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached model and its metadata.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Dictionary with model data or None if not found/expired
        """
        with self.lock:
            if session_id not in self.cache:
                return None
            
            entry = self.cache[session_id]
            
            # Check if expired
            if time.time() - entry['created_at'] > self.ttl_seconds:
                del self.cache[session_id]
                return None
            
            # Update last accessed time
            entry['last_accessed'] = time.time()
            
            return entry
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['created_at'] > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_oldest(self):
        """Remove the least recently accessed entry (LRU eviction)."""
        if not self.cache:
            return
        
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_accessed']
        )
        del self.cache[oldest_key]
    
    def delete_model(self, session_id: str) -> bool:
        """
        Delete a cached model.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if session_id in self.cache:
                del self.cache[session_id]
                return True
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            return {
                'total_models': len(self.cache),
                'max_size': self.max_size,
                'ttl_hours': self.ttl_seconds / 3600,
                'sessions': list(self.cache.keys())
            }


# Global cache instance
_model_cache = ModelCache(max_size=100, ttl_hours=1)


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _model_cache
