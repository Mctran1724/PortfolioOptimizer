"""Caching layer to save and retrieve downloaded financial datasets to local storage."""

import os
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class DataCache:
    """Handles caching of pandas DataFrames to disk (CSV/Parquet) to prevent duplicate API hits."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            # Place it in data/cache relative to workspace
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base_dir, "data", "cache")
            
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_filepath(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.parquet")

    def save(self, df: pd.DataFrame, name: str) -> bool:
        """Saves a DataFrame to cache folder using parquet format."""
        try:
            filepath = self._get_filepath(name)
            df.to_parquet(filepath)
            logger.info(f"Successfully cached dataframe '{name}' to {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache dataframe '{name}': {e}. Attempting CSV fallback.")
            try:
                csv_path = os.path.join(self.cache_dir, f"{name}.csv")
                df.to_csv(csv_path)
                return True
            except Exception as e2:
                logger.error(f"Fallback CSV cache failed: {e2}")
                return False

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Loads a DataFrame from cache. Returns None if it doesn't exist."""
        try:
            filepath = self._get_filepath(name)
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded cached dataframe '{name}' from Parquet.")
                return df
                
            # Try CSV fallback
            csv_path = os.path.join(self.cache_dir, f"{name}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached dataframe '{name}' from CSV.")
                return df
                
            return None
        except Exception as e:
            logger.error(f"Error loading cached dataframe '{name}': {e}")
            return None

    def clear(self) -> None:
        """Clears all cached files in the cache directory."""
        try:
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cache cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
    def get_cache_size_bytes(self) -> int:
        """Returns the size of cache files in bytes."""
        total_size = 0
        if not os.path.exists(self.cache_dir):
            return 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
