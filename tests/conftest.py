import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Application neuve, données dans un dossier temporaire."""
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c
    os.environ.pop("DATASETS_DIR", None)


@pytest.fixture
def upload(client):
    def _upload(content: bytes, name: str = "ventes.csv"):
        return client.post("/datasets/", files={"file": (name, content, "text/csv")})
    return _upload
