"""Core logic: embedding, storage, search."""
from __future__ import annotations

import os
import uuid
import glob
import requests
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance

# Limits
MAX_TEXT_LENGTH = 100_000
MAX_QUERY_LENGTH = 10_000
MAX_EMBED_BATCH = 64
ALLOWED_SCHEMES = {"http", "https"}


class LocalVectorMemory:
    """Local vector memory backed by Ollama embeddings + Qdrant."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        dims: int | None = None,
        db_path: str | None = None,
        collection: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.ollama_url = self._validate_url(
            ollama_url or os.getenv("LVM_OLLAMA_URL", "http://localhost:11434")
        )
        self.model = model or os.getenv("LVM_MODEL", "qwen3-embedding:4b")
        self.dims = dims or int(os.getenv("LVM_DIMS", "2560"))
        self.db_path = db_path or os.getenv("LVM_DB_PATH", "~/.local-vector-memory/qdrant")
        self.collection = collection or os.getenv("LVM_COLLECTION", "memory")
        self.chunk_size = chunk_size or int(os.getenv("LVM_CHUNK_SIZE", "400"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("LVM_CHUNK_OVERLAP", "50"))

        if self.chunk_size < 50 or self.chunk_size > 10000:
            raise ValueError(f"chunk_size must be 50–10000, got {self.chunk_size}")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap must be 0–{self.chunk_size - 1}, got {self.chunk_overlap}")
        if self.dims < 1 or self.dims > 10000:
            raise ValueError(f"dims must be 1–10000, got {self.dims}")

        self.db_path = os.path.expanduser(self.db_path)
        self._client: QdrantClient | None = None

    @staticmethod
    def _validate_url(url: str) -> str:
        """Validate URL to prevent SSRF — must be http(s) to localhost or private IP."""
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            raise ValueError(f"URL scheme must be http/https, got '{parsed.scheme}'")
        if not parsed.hostname:
            raise ValueError("URL must have a hostname")
        # Block non-local hosts (SSRF protection)
        hostname = parsed.hostname.lower()
        allowed = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        if hostname not in allowed and not hostname.endswith(".local") and not hostname.endswith(".localhost"):
            raise ValueError(
                f"Ollama URL must point to localhost (got '{hostname}'). "
                "Set LVM_OLLAMA_URL to a local address."
            )
        return url.rstrip("/")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=self.db_path)
        return self._client

    def init_db(self) -> QdrantClient:
        """Initialize collection if it doesn't exist."""
        c = self.client
        if not c.collection_exists(self.collection):
            c.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dims, distance=Distance.COSINE),
            )
        return c

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama /api/embed, with batch size limit."""
        if len(texts) > MAX_EMBED_BATCH:
            raise ValueError(f"Embed batch too large: {len(texts)} > {MAX_EMBED_BATCH}")
        # Validate individual text lengths
        for t in texts:
            if len(t) > MAX_TEXT_LENGTH:
                raise ValueError(f"Text too long: {len(t)} > {MAX_TEXT_LENGTH} chars")
        r = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"input": texts, "model": self.model},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if len(c.strip()) >= 20]

    def add(self, text: str, source: str = "manual") -> dict:
        """Add a single text entry."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long: {len(text)} > {MAX_TEXT_LENGTH} chars")
        if len(source) > 500:
            raise ValueError("Source label too long")
        c = self.init_db()
        vecs = self.embed([text])
        c.upsert(
            self.collection,
            [PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[0],
                payload={"source": source, "text": text[:2000]},
            )],
        )
        return {"action": "add", "status": "ok", "chunks": 1}

    def search(self, query: str, limit: int = 6) -> list[dict]:
        """Search for similar memories."""
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long: {len(query)} > {MAX_QUERY_LENGTH} chars")
        if limit < 1 or limit > 100:
            raise ValueError(f"Limit must be 1–100, got {limit}")
        c = self.init_db()
        qv = self.embed([query])[0]
        results = c.query_points(
            self.collection, query=qv, limit=limit, with_payload=True
        ).points
        return [
            {
                "score": round(p.score, 4),
                "source": (p.payload or {}).get("source", ""),
                "text": (p.payload or {}).get("text", ""),
            }
            for p in results
        ]

    def stats(self) -> dict:
        """Get collection stats."""
        c = self.client
        if not c.collection_exists(self.collection):
            return {"count": 0, "collection": self.collection}
        info = c.get_collection(self.collection)
        return {
            "collection": self.collection,
            "count": info.points_count or 0,
            "db_path": self.db_path,
        }

    def reindex(
        self,
        directory: str,
        glob_pattern: str = "**/*.md",
        verbose: bool = False,
    ) -> dict:
        """Reindex files from a directory."""
        # Validate glob pattern — no path traversal
        if ".." in glob_pattern:
            raise ValueError("glob pattern must not contain '..'")
        if glob_pattern.startswith("/"):
            raise ValueError("glob pattern must be relative")

        # Resolve and validate directory
        directory = os.path.realpath(os.path.expanduser(directory))

        c = self.init_db()
        # Recreate collection for clean reindex
        if c.collection_exists(self.collection):
            c.delete_collection(self.collection)
        c.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.dims, distance=Distance.COSINE),
        )

        files = sorted(glob.glob(os.path.join(directory, glob_pattern), recursive=True))
        total_chunks = 0

        for fpath in files:
            # Verify resolved path is still under directory (no symlink escape)
            real_path = os.path.realpath(fpath)
            if not real_path.startswith(directory):
                if verbose:
                    print(f"  ⚠️ Skipping (path escape): {fpath}")
                continue

            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
            except (PermissionError, OSError):
                continue
            if len(content) < 50:
                continue

            rel = os.path.relpath(fpath, directory)
            chunks = self._chunk_text(content)
            if not chunks:
                continue

            # Embed in batches
            for batch_start in range(0, len(chunks), MAX_EMBED_BATCH):
                batch = chunks[batch_start:batch_start + MAX_EMBED_BATCH]
                vecs = self.embed(batch)
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=v,
                        payload={"source": rel, "chunk": batch_start + i, "text": batch[i]},
                    )
                    for i, v in enumerate(vecs)
                ]
                c.upsert(self.collection, points)
            total_chunks += len(chunks)
            if verbose:
                print(f"  ✅ {rel} [{len(chunks)} chunks]")

        return {
            "action": "reindex",
            "files": len(files),
            "total_chunks": total_chunks,
        }

    def delete_source(self, source: str) -> dict:
        """Delete all points matching a source."""
        if len(source) > 500:
            raise ValueError("Source label too long")
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        c = self.client
        c.delete(
            self.collection,
            filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        return {"action": "delete", "source": source, "status": "ok"}
