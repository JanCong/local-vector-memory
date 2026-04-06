"""Tests for local_vector_memory."""
from __future__ import annotations

import os
import json
import tempfile
import pytest

from local_vector_memory.core import LocalVectorMemory, MAX_TEXT_LENGTH, MAX_QUERY_LENGTH


# ── Fixtures ──

@pytest.fixture
def lvm(tmp_path):
    """Create an LVM instance with a temp DB path."""
    return LocalVectorMemory(
        ollama_url="http://localhost:11434",
        db_path=str(tmp_path / "qdrant"),
        collection="test",
    )


@pytest.fixture
def lvm_with_data(lvm, tmp_path):
    """Create LVM with some test files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note1.md").write_text("# Test\nThis is a test note about machine learning and AI.\n" * 10)
    (docs / "note2.md").write_text("# Python\nPython is a great programming language for data science.\n" * 10)
    return lvm, docs


# ── SSRF Protection ──

class TestSSRFProtection:
    def test_blocks_aws_metadata(self):
        with pytest.raises(ValueError, match="localhost"):
            LocalVectorMemory(ollama_url="http://169.254.169.254/latest/")

    def test_blocks_remote_host(self):
        with pytest.raises(ValueError, match="localhost"):
            LocalVectorMemory(ollama_url="http://evil.com/api/")

    def test_allows_localhost(self):
        lvm = LocalVectorMemory(ollama_url="http://localhost:11434")
        assert lvm.ollama_url == "http://localhost:11434"

    def test_allows_127(self):
        lvm = LocalVectorMemory(ollama_url="http://127.0.0.1:11434")
        assert lvm.ollama_url == "http://127.0.0.1:11434"

    def test_allows_dot_local(self):
        lvm = LocalVectorMemory(ollama_url="http://my-server.local:11434")
        assert lvm.ollama_url == "http://my-server.local:11434"

    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="scheme"):
            LocalVectorMemory(ollama_url="ftp://localhost/")

    def test_strips_trailing_slash(self):
        lvm = LocalVectorMemory(ollama_url="http://localhost:11434/")
        assert lvm.ollama_url == "http://localhost:11434"


# ── Input Validation ──

class TestInputValidation:
    def test_text_too_long(self, lvm):
        with pytest.raises(ValueError, match="too long"):
            lvm.add("x" * (MAX_TEXT_LENGTH + 1))

    def test_source_too_long(self, lvm):
        with pytest.raises(ValueError, match="too long"):
            lvm.add("hello", source="s" * 501)

    def test_query_too_long(self, lvm):
        with pytest.raises(ValueError, match="too long"):
            lvm.search("q" * (MAX_QUERY_LENGTH + 1))

    def test_limit_range(self, lvm):
        with pytest.raises(ValueError, match="1–100"):
            lvm.search("test", limit=0)
        with pytest.raises(ValueError, match="1–100"):
            lvm.search("test", limit=101)

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="dims"):
            LocalVectorMemory(dims=0, ollama_url="http://localhost:11434")

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            LocalVectorMemory(chunk_size=10)

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            LocalVectorMemory(chunk_size=400, chunk_overlap=400)


# ── Path Traversal ──

class TestPathTraversal:
    def test_blocks_dotdot_glob(self, lvm):
        with pytest.raises(ValueError, match="\\.\\."):
            lvm.reindex("/tmp", glob_pattern="../../etc/**/*.md")

    def test_blocks_absolute_glob(self, lvm):
        with pytest.raises(ValueError, match="relative"):
            lvm.reindex("/tmp", glob_pattern="/etc/passwd")


# ── Core Logic (unit, no Ollama needed) ──

class TestChunking:
    def test_basic_chunking(self, lvm):
        text = "A" * 1000
        chunks = lvm._chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= lvm.chunk_size for c in chunks)
        # Check overlap
        if len(chunks) > 1:
            overlap = chunks[0][lvm.chunk_size - lvm.chunk_overlap:]
            assert chunks[1][:lvm.chunk_overlap] == overlap

    def test_short_text_filtered(self, lvm):
        chunks = lvm._chunk_text("short")
        assert chunks == []

    def test_exact_chunk_size(self, lvm):
        text = "A" * 400
        chunks = lvm._chunk_text(text)
        # chunk_size=400, overlap=50 → second chunk starts at 350, gets 50 chars
        assert len(chunks) == 2
        assert len(chunks[0]) == 400


class TestInitDB:
    def test_creates_collection(self, lvm):
        c = lvm.init_db()
        assert c.collection_exists("test")

    def test_idempotent(self, lvm):
        lvm.init_db()
        lvm.init_db()  # should not raise
        assert lvm.client.collection_exists("test")


class TestStats:
    def test_empty_collection(self, lvm):
        stats = lvm.stats()
        assert stats["count"] == 0
        assert stats["collection"] == "test"

    def test_after_init(self, lvm):
        lvm.init_db()
        stats = lvm.stats()
        assert stats["count"] == 0


# ── CLI Tests ──

class TestCLI:
    def test_no_args_shows_help(self, capsys):
        from local_vector_memory.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version(self, capsys):
        from local_vector_memory.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert "0.1.0" in capsys.readouterr().out

    def test_search_too_long_query(self, capsys):
        from local_vector_memory.cli import main
        with pytest.raises(ValueError, match="too long"):
            main(["search", "x" * (MAX_QUERY_LENGTH + 1)])
