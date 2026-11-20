import json
import os
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")  # rendu sans affichage, côté serveur

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATASETS_DIR", "datasets"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE = DATA_DIR / "datasets.json"

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_COLUMNS_TO_PLOT = 10
MAX_SAMPLE_SIZE = 10000
PLOTS_PER_PAGE = 6  # grille 2 x 3

app = FastAPI(title="Dataset API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/app/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# Matplotlib n'est pas thread-safe : une seule génération de PDF à la fois.
plot_lock = threading.Lock()


def load_datasets() -> Dict[str, dict]:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def save_datasets(datasets: Dict[str, dict]):
    METADATA_FILE.write_text(json.dumps(datasets, indent=2, ensure_ascii=False), encoding="utf-8")


datasets = load_datasets()


def get_dataset_or_404(dataset_id: str) -> dict:
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets[dataset_id]


def read_dataset(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{dataset_id}.csv")


def numerical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def attachment(filename: str) -> Dict[str, str]:
    """En-tête de téléchargement qui supporte les noms de fichiers accentués (RFC 6266)."""
    ascii_name = filename.encode("ascii", "replace").decode().replace('"', "")
    return {"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"}


def get_top_columns_by_variance(df: pd.DataFrame, max_cols: int = MAX_COLUMNS_TO_PLOT) -> List[str]:
    """Les N colonnes numériques de plus forte variance."""
    cols = numerical_columns(df)
    if len(cols) <= max_cols:
        return cols
    return df[cols].var().nlargest(max_cols).index.tolist()


def create_subplot_grid(n_plots: int, max_cols: int = 2) -> Tuple[int, int]:
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    return n_rows, n_cols


def _grid_page(pdf, df, cols, title, draw):
    n_rows, n_cols = create_subplot_grid(len(cols))
    fig = plt.figure(figsize=(15, 5 * n_rows))
    fig.suptitle(title, fontsize=16, y=1.02)
    for idx, col in enumerate(cols, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        draw(ax, col)
        ax.tick_params(axis="both", which="major", labelsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _histogram(df):
    def draw(ax, col):
        sns.histplot(data=df, x=col, kde=df[col].nunique() > 1, ax=ax)
        ax.axvline(df[col].mean(), color="red", linestyle="--", alpha=0.8, label=f"Mean: {df[col].mean():.2f}")
        ax.axvline(df[col].median(), color="green", linestyle="--", alpha=0.8,
                   label=f"Median: {df[col].median():.2f}")
        ax.set_title(f"Distribution of {col}", pad=20, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
    return draw


def generate_visualizations(df: pd.DataFrame) -> BytesIO:
    """Rapport PDF en mémoire : histogrammes, boîtes, courbes, pairplot, corrélations."""
    if len(df) > MAX_SAMPLE_SIZE:
        df = df.sample(n=MAX_SAMPLE_SIZE, random_state=42)
    cols = get_top_columns_by_variance(df)
    output = BytesIO()
    with plot_lock:
        try:
            with PdfPages(output) as pdf:
                for i in range(0, len(cols), PLOTS_PER_PAGE):
                    batch = cols[i:i + PLOTS_PER_PAGE]
                    _grid_page(pdf, df, batch, "Histograms with KDE", _histogram(df))
                    _grid_page(pdf, df, batch, "Box Plots",
                               lambda ax, col: (sns.boxplot(data=df, y=col, ax=ax),
                                                ax.set_title(f"Box Plot of {col}", pad=20, fontsize=12)))
                    _grid_page(pdf, df, batch, "Value vs Index Line Plots",
                               lambda ax, col: (sns.lineplot(x=np.arange(len(df)), y=df[col].to_numpy(), ax=ax),
                                                ax.set_title(f"Line Plot of {col}", pad=20, fontsize=12),
                                                ax.set_xlabel("Index", fontsize=10)))
                # Pairplot et corrélations : sur le premier lot (les colonnes de plus forte variance).
                first = cols[:PLOTS_PER_PAGE]
                if len(first) > 1:
                    grid = sns.pairplot(df[first].dropna(), diag_kind="kde")
                    grid.fig.suptitle("Scatterplot Matrix", y=1.02, fontsize=16)
                    pdf.savefig(grid.fig, bbox_inches="tight")
                    plt.close(grid.fig)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(df[first].corr(), annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
                    ax.set_title("Correlation Heatmap", pad=20, fontsize=16)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        finally:
            plt.close("all")
    output.seek(0)
    return output


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.post("/datasets/")
async def create_dataset(file: UploadFile = File(...)):
    """Importe un fichier CSV et enregistre ses métadonnées."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        df = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable CSV file: {exc}") from exc

    dataset_id = str(uuid.uuid4())
    (DATA_DIR / f"{dataset_id}.csv").write_bytes(content)
    datasets[dataset_id] = {
        "id": dataset_id,
        "filename": file.filename,
        "size": len(content),
        "rows": len(df),
        "columns": len(df.columns),
    }
    save_datasets(datasets)
    return datasets[dataset_id]


@app.get("/datasets/")
async def list_datasets():
    return list(datasets.values())


@app.get("/datasets/{dataset_id}/")
async def get_dataset(dataset_id: str):
    return get_dataset_or_404(dataset_id)


@app.delete("/datasets/{dataset_id}/")
async def delete_dataset(dataset_id: str):
    get_dataset_or_404(dataset_id)
    (DATA_DIR / f"{dataset_id}.csv").unlink(missing_ok=True)
    del datasets[dataset_id]
    save_datasets(datasets)
    return {"message": "Dataset deleted successfully"}


@app.get("/datasets/{dataset_id}/excel/")
async def export_excel(dataset_id: str):
    meta = get_dataset_or_404(dataset_id)
    output = BytesIO()
    read_dataset(dataset_id).to_excel(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=attachment(f"{Path(meta['filename']).stem}.xlsx"),
    )


@app.get("/datasets/{dataset_id}/stats/")
async def get_stats(dataset_id: str):
    """Statistiques descriptives (count, mean, std, min, quartiles, max) par colonne."""
    get_dataset_or_404(dataset_id)
    stats = read_dataset(dataset_id).describe()
    # NaN n'existe pas en JSON : on le remplace par null.
    return json.loads(stats.to_json(double_precision=15))


@app.get("/datasets/{dataset_id}/plot/")
async def generate_plots(dataset_id: str):
    meta = get_dataset_or_404(dataset_id)
    df = read_dataset(dataset_id)
    if not numerical_columns(df):
        raise HTTPException(status_code=400, detail="No numerical columns found in dataset")
    pdf_bytes = await run_in_threadpool(generate_visualizations, df)
    return StreamingResponse(pdf_bytes, media_type="application/pdf",
                             headers=attachment(f"{Path(meta['filename']).stem}_plots.pdf"))
