"""
Simple Generic Cache for Investigation C

Unlike Investigation B's CacheManager which has specific methods for
ODE generation workflows, this provides generic key-value caching.
"""

import os
import pickle
import hashlib
from pathlib import Path


class SimpleCache:
    def __init__(self, cache_dir='.cache_investigation_c'):
        """
        Initialize simple cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.hits = 0
        self.misses = 0
    
    def _get_path(self, key):
        """Get cache file path for key"""
        # Hash the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def exists(self, key):
        """Check if key exists in cache"""
        return self._get_path(key).exists()
    
    def load(self, key):
        """
        Load value from cache
        
        Args:
            key: Cache key string
        
        Returns:
            Cached value
        
        Raises:
            FileNotFoundError: If key not in cache
        """
        cache_path = self._get_path(key)
        
        if not cache_path.exists():
            self.misses += 1
            raise FileNotFoundError(f"Cache key not found: {key}")
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            self.hits += 1
            return value
        except Exception as e:
            self.misses += 1
            raise RuntimeError(f"Failed to load cache: {e}")
    
    def save(self, key, value):
        """
        Save value to cache
        
        Args:
            key: Cache key string
            value: Value to cache (must be picklable)
        """
        cache_path = self._get_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
    
    def clear(self):
        """Clear all cached data"""
        count = 0
        for f in self.cache_dir.glob('*.pkl'):
            f.unlink()
            count += 1
        if count > 0:
            print(f"✓ Cleared cache ({count} files)")
    
    def report(self):
        """Print cache statistics"""
        files = list(self.cache_dir.glob('*.pkl'))
        size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        print("\n" + "="*60)
        print("CACHE STATISTICS")
        print("="*60)
        print(f"Files:      {len(files)}")
        print(f"Size:       {size_mb:.2f} MB")
        print(f"Hits:       {self.hits}/{total_requests} ({hit_rate:.1f}%)")
        print(f"Location:   {self.cache_dir.absolute()}")
        print("="*60)