from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import uuid
from pathlib import Path
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from fastapi.concurrency import run_in_threadpool
import numpy as np
from warnings import warn
import threading
import atexit

app = FastAPI(title="Dataset API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create data directory if it doesn't exist
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

# File to store dataset metadata
METADATA_FILE = DATA_DIR / "datasets.json"

# Constants for visualization
MAX_COLUMNS_TO_PLOT = 10
MAX_SAMPLE_SIZE = 10000
PLOTS_PER_PAGE = 6  # 2x3 grid

# Ensure matplotlib is properly configured
plt.ioff()  # Turn off interactive mode
plt.switch_backend('Agg')  # Ensure we're using the Agg backend

def cleanup_matplotlib():
    """Cleanup function to be called on exit"""
    plt.close('all')

# Register cleanup function
atexit.register(cleanup_matplotlib)

# Load existing datasets if available
def load_datasets() -> Dict[str, dict]:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save datasets to file
def save_datasets(datasets: Dict[str, dict]):
    with open(METADATA_FILE, 'w') as f:
        json.dump(datasets, f, indent=2)

# Initialize datasets
datasets = load_datasets()

def get_top_columns_by_variance(df: pd.DataFrame, max_cols: int = MAX_COLUMNS_TO_PLOT) -> List[str]:
    """Get top N numerical columns by variance"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) <= max_cols:
        return numerical_cols.tolist()
    
    variances = df[numerical_cols].var()
    return variances.nlargest(max_cols).index.tolist()

def create_subplot_grid(n_plots: int, max_cols: int = 2) -> Tuple[int, int]:
    """Create optimal subplot grid dimensions"""
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    return n_rows, n_cols

def generate_visualizations(df: pd.DataFrame, numerical_cols: List[str]) -> BytesIO:
    """Generate all visualizations and return as PDF in memory"""
    output = BytesIO()
    
    # Sample large datasets
    if len(df) > MAX_SAMPLE_SIZE:
        df = df.sample(n=MAX_SAMPLE_SIZE, random_state=42)
        warn(f"Dataset sampled to {MAX_SAMPLE_SIZE} rows for visualization")
    
    # Limit number of columns
    if len(numerical_cols) > MAX_COLUMNS_TO_PLOT:
        numerical_cols = get_top_columns_by_variance(df)
        warn(f"Limited to top {MAX_COLUMNS_TO_PLOT} columns by variance")
    
    try:
        with PdfPages(output) as pdf:
            # Process columns in batches
            for i in range(0, len(numerical_cols), PLOTS_PER_PAGE):
                batch_cols = numerical_cols[i:i + PLOTS_PER_PAGE]
                n_rows, n_cols = create_subplot_grid(len(batch_cols), max_cols=2)
                
                # 1. Histograms with KDE
                fig = plt.figure(figsize=(15, 5 * n_rows))
                fig.suptitle('Histograms with KDE', fontsize=16, y=1.02)
                
                for idx, col in enumerate(batch_cols, 1):
                    ax = plt.subplot(n_rows, n_cols, idx)
                    sns.histplot(data=df, x=col, kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}', pad=20, fontsize=12)
                    ax.set_xlabel(col, fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                    ax.legend(fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
                # 2. Box Plots
                fig = plt.figure(figsize=(15, 5 * n_rows))
                fig.suptitle('Box Plots', fontsize=16, y=1.02)
                
                for idx, col in enumerate(batch_cols, 1):
                    ax = plt.subplot(n_rows, n_cols, idx)
                    sns.boxplot(data=df, y=col, ax=ax)
                    ax.set_title(f'Box Plot of {col}', pad=20, fontsize=12)
                    ax.set_ylabel(col, fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
                # 3. Value-vs-Index Line Plots
                fig = plt.figure(figsize=(15, 5 * n_rows))
                fig.suptitle('Value vs Index Line Plots', fontsize=16, y=1.02)
                
                for idx, col in enumerate(batch_cols, 1):
                    ax = plt.subplot(n_rows, n_cols, idx)
                    sns.lineplot(data=df, x=df.index, y=col, ax=ax)
                    ax.set_title(f'Line Plot of {col}', pad=20, fontsize=12)
                    ax.set_xlabel('Index', fontsize=10)
                    ax.set_ylabel(col, fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            
            # 4. Scatterplot Matrix (Pairplot) - only for first batch if multiple batches
            if len(numerical_cols) > 1:
                fig = sns.pairplot(df[batch_cols], diag_kind='kde')
                fig.fig.suptitle('Scatterplot Matrix', y=1.02, fontsize=16)
                plt.tight_layout()
                pdf.savefig(fig.fig)
                plt.close(fig.fig)
            
            # 5. Correlation Heatmap - only for first batch if multiple batches
            if len(numerical_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[batch_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
                ax.set_title('Correlation Heatmap', pad=20, fontsize=16)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    finally:
        # Ensure all figures are closed
        plt.close('all')
    
    output.seek(0)
    return output

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return FileResponse("app/static/index.html")

@app.post("/datasets/")
async def create_dataset(file: UploadFile = File(...)):
    """Create a new dataset from uploaded CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique ID for the dataset
    dataset_id = str(uuid.uuid4())
    
    # Save the file
    file_path = DATA_DIR / f"{dataset_id}.csv"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Read the CSV to get basic info
    df = pd.read_csv(file_path)
    
    # Store metadata
    datasets[dataset_id] = {
        "id": dataset_id,
        "filename": file.filename,
        "size": len(content),
        "rows": len(df),
        "columns": len(df.columns)
    }
    
    # Save metadata to file
    save_datasets(datasets)
    
    return datasets[dataset_id]

@app.get("/datasets/")
async def list_datasets():
    """List all uploaded datasets"""
    return list(datasets.values())

@app.get("/datasets/{dataset_id}/")
async def get_dataset(dataset_id: str):
    """Get dataset information"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets[dataset_id]

@app.delete("/datasets/{dataset_id}/")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete the file
    file_path = DATA_DIR / f"{dataset_id}.csv"
    if file_path.exists():
        file_path.unlink()
    
    # Remove from metadata
    del datasets[dataset_id]
    
    # Save updated metadata
    save_datasets(datasets)
    
    return {"message": "Dataset deleted successfully"}

@app.get("/datasets/{dataset_id}/excel/")
async def export_excel(dataset_id: str):
    """Export dataset to Excel"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    file_path = DATA_DIR / f"{dataset_id}.csv"
    excel_path = DATA_DIR / f"{dataset_id}.xlsx"
    
    df = pd.read_csv(file_path)
    df.to_excel(excel_path, index=False)
    
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{datasets[dataset_id]['filename']}.xlsx"
    )

@app.get("/datasets/{dataset_id}/stats/")
async def get_stats(dataset_id: str):
    """Get dataset statistics"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    file_path = DATA_DIR / f"{dataset_id}.csv"
    df = pd.read_csv(file_path)
    
    # Get statistics for numerical columns
    stats = df.describe().to_dict()
    return JSONResponse(content=stats)

@app.get("/datasets/{dataset_id}/plot/")
async def generate_plots(dataset_id: str):
    """Generate comprehensive visualizations for the dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    file_path = DATA_DIR / f"{dataset_id}.csv"
    
    # Read CSV and generate plots in a separate thread
    async def process_data():
        df = pd.read_csv(file_path)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) == 0:
            raise HTTPException(status_code=400, detail="No numerical columns found in dataset")
        
        # Generate visualizations in a separate thread
        pdf_bytes = await run_in_threadpool(generate_visualizations, df, numerical_cols)
        return pdf_bytes
    
    pdf_bytes = await process_data()
    
    return StreamingResponse(
        pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{datasets[dataset_id]["filename"]}_plots.pdf"'
        }
    ) 
