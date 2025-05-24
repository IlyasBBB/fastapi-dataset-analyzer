from fastapi import FastAPI
from pathlib import Path

app = FastAPI(title="Dataset API")

# Create data directory if it doesn't exist
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to Dataset API"}
