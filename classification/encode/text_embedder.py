""" Embeddings encoder for text sequences
    Uses SentenceTransformers, but adds a cache
"""
import hashlib
import json
import os

import numpy
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Embed some text
    """

    def __init__(self, model: str, cache_file: str | None):
        """
        Initialize the encoder, based on sentence transformers
        :param model: The model to load
        :param cache_file: The file used to keep the cache
        """
        self.cache_file = cache_file
        if not self.cache_file:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            self.cache_file = os.path.join(cache_dir, "embedding_cache.json")
        # The cache, loaded at initialization
        self.cache = {}
        # New cache items created at runtime
        self.cache_new = []
        if self.cache_file:
            self._load_cache()
        self.model = model
        self.sentence_transformer = SentenceTransformer(model)

    def _load_cache(self) -> None:
        """Load cached embeddings from the JSONL cache file."""
        if not os.path.exists(self.cache_file):
            open(self.cache_file, "w").close()
        with open(self.cache_file, "r") as cache_data:
            for line in cache_data:
                record = json.loads(line)
                md5 = record["md5"]
                model = record["model"]
                embedding = numpy.array(record["embedding"], dtype=float)
                self.cache[(md5, model)] = embedding

    def _save_cache(self) -> None:
        """Append newly computed embeddings to the cache file."""
        with open(self.cache_file, "a") as cache_file:
            for md5, model, embedding in self.cache_new:
                record = {
                    "md5": md5,
                    "model": model,
                    "embedding": numpy.around(embedding, decimals=5).tolist(),
                }
                # Dump to json; save some space by removing unnecessary blanks
                json_dump = json.dumps(record).replace(", ", ",")
                cache_file.write(json_dump + "\n")
            self.cache_new = []

    def encode(self, texts: list[str]) -> numpy.ndarray:
        """
        Encode a list of texts into embedding vectors, using cache where available.

        :param texts: List of text strings to encode.
        :return: Array of shape (len(texts), embedding_dim).
        """
        embeddings = []
        for text in texts:
            md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
            embedding = self.cache.get((md5, self.model), None)
            if embedding is None:
                embedding = self.sentence_transformer.encode([text])[0]
                embedding = numpy.array(embedding, dtype=float)
                self.cache[(md5, self.model)] = embedding
                self.cache_new.append((md5, self.model, embedding))
                batch_size = 200
                if len(self.cache) % batch_size == 0:
                    print(
                        f"INFO: Embedding calculation - next batch of {batch_size} embeddings of {len(texts)} completed"
                    )
                    self._save_cache()
            embeddings.append([embedding])
            if len(embeddings) % 1000 == 0:
                print(
                    f"INFO: Embedding retrieval/calculation - {len(embeddings)} of {len(texts)} completed"
                )
                self._save_cache()
        return numpy.concatenate(embeddings)
