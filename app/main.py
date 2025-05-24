from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from warnings import warn

app = FastAPI(title="Dataset API")

# Create data directory if it doesn't exist
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# In-memory store of metadata
datasets = {}

# Constants for enhanced visualizations
MAX_SAMPLE_SIZE = 10_000
MAX_COLUMNS_TO_PLOT = 10
PLOTS_PER_PAGE = 6  # Maximum number of subplots per PDF page (2x3 grid)

def get_top_columns_by_variance(df: pd.DataFrame, max_cols: int = MAX_COLUMNS_TO_PLOT) -> list[str]:
    """Get up to `max_cols` numeric columns with the highest variance."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) <= max_cols:
        return numerical_cols.tolist()
    variances = df[numerical_cols].var()
    return variances.nlargest(max_cols).index.tolist()

def create_subplot_grid(n_plots: int, max_cols: int = 2) -> tuple[int, int]:
    """Compute the number of rows and columns for a grid of `n_plots`."""
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    return n_rows, n_cols

def generate_visualizations(df: pd.DataFrame, numerical_cols: list[str], pdf_path: Path):
    """
    Generate enhanced visualizations and save them into a multi-page PDF:
      1. Histograms with KDE
      2. Box plots
      3. Line plots (value vs. index)
      4. Scatterplot matrix (pairplot) for the first batch
      5. Correlation heatmap for the first batch

    Applies sampling if the dataset is large and caps the number of columns by variance.
    """
    # 1. Sample large datasets
    if len(df) > MAX_SAMPLE_SIZE:
        df = df.sample(n=MAX_SAMPLE_SIZE, random_state=42)
        warn(f"Dataset sampled to {MAX_SAMPLE_SIZE} rows for visualization")

    # 2. Cap number of columns
    if len(numerical_cols) > MAX_COLUMNS_TO_PLOT:
        numerical_cols = get_top_columns_by_variance(df)
        warn(f"Limited to top {MAX_COLUMNS_TO_PLOT} numeric columns by variance")

    with PdfPages(pdf_path) as pdf:
        # Process columns in batches
        for start in range(0, len(numerical_cols), PLOTS_PER_PAGE):
            batch = numerical_cols[start : start + PLOTS_PER_PAGE]
            n_rows, n_cols = create_subplot_grid(len(batch))

            # a) Histograms + KDE
            fig = plt.figure(figsize=(8 * n_cols, 4 * n_rows))
            fig.suptitle('Histograms with KDE', fontsize=16, y=1.02)
            for idx, col in enumerate(batch, start=1):
                ax = fig.add_subplot(n_rows, n_cols, idx)
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
                ax.axvline(df[col].median(), color='green', linestyle='--', label='Median')
                ax.legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # b) Box Plots
            fig = plt.figure(figsize=(8 * n_cols, 4 * n_rows))
            fig.suptitle('Box Plots', fontsize=16, y=1.02)
            for idx, col in enumerate(batch, start=1):
                ax = fig.add_subplot(n_rows, n_cols, idx)
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                ax.set_ylabel(col)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # c) Line Plots (Value vs. Index)
            fig = plt.figure(figsize=(8 * n_cols, 4 * n_rows))
            fig.suptitle('Value vs. Index Line Plots', fontsize=16, y=1.02)
            for idx, col in enumerate(batch, start=1):
                ax = fig.add_subplot(n_rows, n_cols, idx)
                sns.lineplot(x=df.index, y=df[col], ax=ax)
                ax.set_title(f'Line Plot of {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel(col)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # d) Scatterplot Matrix (Pairplot) - first batch only
        if len(numerical_cols) > 1:
            first_batch = numerical_cols[:PLOTS_PER_PAGE]
            pair = sns.pairplot(df[first_batch], diag_kind='kde')
            pair.fig.suptitle('Scatterplot Matrix', y=1.02, fontsize=16)
            pdf.savefig(pair.fig)
            plt.close(pair.fig)

            # e) Correlation Heatmap - first batch only
            corr = df[first_batch].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap', pad=20, fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

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

@app.get("/datasets/{dataset_id}/stats/")
async def get_stats(dataset_id: str):
    """Return basic statistics (describe) for a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    csv_path = DATA_DIR / f"{dataset_id}.csv"
    df = pd.read_csv(csv_path)
    # Return describe() summary as JSON-friendly dict
    stats = df.describe().to_dict()
    return stats

@app.get("/datasets/{dataset_id}/plot/")
async def generate_plots(dataset_id: str):
    """Generate comprehensive visualizations for the dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    csv_path = DATA_DIR / f"{dataset_id}.csv"
    df = pd.read_csv(csv_path)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="No numeric columns found in dataset")

    # Generate and save all plots in batches
    pdf_path = DATA_DIR / f"{dataset_id}.pdf"
    generate_visualizations(df, numeric_cols, pdf_path)

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{datasets[dataset_id]['filename']}_plots.pdf"
    )
