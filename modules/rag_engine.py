"""Retrieval-Augmented Generation engine: PDF/text -> chunks -> embeddings -> FAISS."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

SUPPORTED = {".pdf", ".txt", ".md"}


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("। ", start, end))
            if cut > start + size * 0.5:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 40]


def extract_pages(name: str, data: bytes):
    """Return a list of (page_number, text)."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            return [(i + 1, p.get_text()) for i, p in enumerate(doc)]
    return [(1, data.decode("utf-8", errors="ignore"))]


class RAGEngine:
    def __init__(self, docs_dir=config.DOCS_DIR, index_dir=config.VECTOR_DIR,
                 model_name=config.EMBED_MODEL):
        self.docs_dir, self.index_dir, self.model_name = Path(docs_dir), Path(index_dir), model_name
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "meta.json"
        self._model = None
        self.index = None
        self.meta = []
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            import faiss
            self.index = faiss.read_index(str(self.index_path))
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self):
        import faiss
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")

    # ---------- embeddings ----------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts):
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    # ---------- indexing ----------
    def _add_chunks(self, records):
        import faiss
        if not records:
            return 0
        emb = self._embed([r["text"] for r in records])
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity (normalized vectors)
        self.index.add(emb)
        self.meta.extend(records)
        self._save()
        return len(records)

    def _records_for(self, name: str, data: bytes):
        recs = []
        for page, text in extract_pages(name, data):
            for c in chunk_text(text):
                recs.append({"text": c, "source": name, "page": page})
        return recs

    def build(self):
        """Rebuild the whole index from files in docs_dir."""
        self.index, self.meta = None, []
        total = 0
        for p in sorted(self.docs_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED:
                total += self._add_chunks(self._records_for(p.name, p.read_bytes()))
        return total

    def add_document(self, name: str, data: bytes) -> int:
        """Save an uploaded document and add it to the index (replaces the same filename)."""
        (self.docs_dir / name).write_bytes(data)
        if any(m["source"] == name for m in self.meta):
            return self.build()
        return self._add_chunks(self._records_for(name, data))

    # ---------- retrieval ----------
    def search(self, query: str, k: int = config.TOP_K):
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(self._embed([query]), min(k, self.index.ntotal))
        return [dict(self.meta[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def stats(self) -> pd.DataFrame:
        if not self.meta:
            return pd.DataFrame(columns=["source", "chunks"])
        df = pd.DataFrame(self.meta).groupby("source").size().reset_index(name="chunks")
        return df.sort_values("chunks", ascending=False)
