from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import pandas as pd

app = FastAPI(title="Dataset API")

# Create data directory if it doesn't exist
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# In‐memory store of metadata
datasets = {}

@app.get("/")
async def root():
    return {"message": "Welcome to Dataset API"}

@app.post("/datasets/")
async def create_dataset(file: UploadFile = File(...)):
    """Upload a CSV, save it, load into pandas, & return its ID."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    # 1. Assign a new UUID
    dataset_id = str(uuid.uuid4())
    # 2. Persist the bytes to disk
    file_path = DATA_DIR / f"{dataset_id}.csv"
    contents = await file.read()
    file_path.write_bytes(contents)
    # 3. Load with pandas to inspect
    df = pd.read_csv(file_path)
    # 4. Store metadata in memory
    datasets[dataset_id] = {
        "id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns)
    }
    return datasets[dataset_id]
