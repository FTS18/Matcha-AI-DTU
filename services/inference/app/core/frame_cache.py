import hashlib
import numpy as np
from collections import deque


class FrameCache:
    """LRU cache for YOLO inference results to avoid redundant processing."""

    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.95):
        self.cache = {}
        self.order = deque(maxlen=max_size)
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def _get_hash(self, frame: np.ndarray) -> str:
        """Compute frame hash using the first few bytes."""
        # Sample every 4th pixel for speed
        return hashlib.md5(frame[::4, ::4].tobytes()).hexdigest()

    def get(self, frame: np.ndarray):
        """Try to retrieve cached result for similar frame."""
        frame_hash = self._get_hash(frame)
        if frame_hash in self.cache:
            self.hits += 1
            return self.cache[frame_hash]
        self.misses += 1
        return None

    def put(self, frame: np.ndarray, result):
        """Cache the inference result."""
        frame_hash = self._get_hash(frame)
        if len(self.order) >= self.max_size and frame_hash not in self.cache:
            oldest = self.order[0]
            del self.cache[oldest]
        self.cache[frame_hash] = result
        self.order.append(frame_hash)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache),
        }
