from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
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

@app.get("/datasets/")
async def list_datasets():
    """List metadata for all uploaded datasets"""
    return list(datasets.values())

@app.get("/datasets/{dataset_id}/")
async def get_dataset(dataset_id: str):
    """Retrieve metadata for a single dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets[dataset_id]

@app.delete("/datasets/{dataset_id}/")
async def delete_dataset(dataset_id: str):
    """Delete a dataset file and its metadata entry"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Remove file from disk
    file_path = DATA_DIR / f"{dataset_id}.csv"
    if file_path.exists():
        file_path.unlink()
    # Remove metadata entry
    del datasets[dataset_id]
    return {"message": "Dataset deleted successfully"}

@app.get("/datasets/{dataset_id}/excel/")
async def export_excel(dataset_id: str):
    """Export dataset to Excel"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    csv_path = DATA_DIR / f"{dataset_id}.csv"
    df = pd.read_csv(csv_path)
    xlsx_path = DATA_DIR / f"{dataset_id}.xlsx"
    df.to_excel(xlsx_path, index=False)
    return FileResponse(
        xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{datasets[dataset_id]['filename']}.xlsx"
    )