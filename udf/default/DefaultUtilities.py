import os

EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "cache/embeddings")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings")
