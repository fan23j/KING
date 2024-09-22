import functools
from typing import Dict, Tuple

class DimensionsCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DimensionsCache, cls).__new__(cls)
            cls._instance.cache = {}
        return cls._instance
    
    def set(self, key: str, value: Tuple[float, Dict, Dict]) -> None:
        self.cache[key] = value
    
    def get(self, key: str) -> Tuple[float, Dict, Dict]:
        return self.cache.get(key)
    
    def has(self, key: str) -> bool:
        return key in self.cache
    
    def clear(self) -> None:
        self.cache.clear()

def cache_dimensions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache = DimensionsCache()
        json_dir = kwargs.get('json_dir') or args[0]
        cached_result = cache.get(json_dir)
        if cached_result is not None:
            return cached_result
        result = func(*args, **kwargs)
        cache.set(json_dir, result)
        return result
    return wrapper