# fastapi-dataset-analyzer

A FastAPI-based service for uploading, analyzing, and visualizing datasets. This project provides a RESTful API for managing CSV datasets and generating comprehensive visualizations.

## Features

### Dataset Management

- Upload CSV datasets with automatic metadata extraction
- List all available datasets with their metadata
- Get detailed information about specific datasets
- Delete datasets when no longer needed
- Export datasets to Excel format

### Statistical Analysis

- Generate comprehensive statistical summaries including:
  - Basic statistics (mean, median, standard deviation)
  - Data distribution metrics
  - Column-wise statistics
  - Data quality metrics

### Advanced Visualizations

The API generates a comprehensive PDF report containing multiple visualizations:

1. **Histograms with KDE (Kernel Density Estimation)**

   - Distribution analysis for each numerical column
   - Mean and median indicators
   - KDE overlay for better distribution understanding

2. **Box Plots**

   - Outlier detection
   - Quartile analysis
   - Distribution spread visualization

3. **Value-vs-Index Line Plots**

   - Trend analysis
   - Time series visualization
   - Pattern identification

4. **Scatterplot Matrix (Pairplot)**

   - Multi-variable correlation visualization
   - Distribution analysis
   - Relationship patterns between variables

5. **Correlation Heatmap**
   - Inter-variable correlation analysis
   - Correlation coefficient visualization
   - Pattern identification in relationships

### Performance Optimizations

- Asynchronous processing for better performance
- Automatic sampling for large datasets (max 10,000 rows for visualization)
- Smart column selection based on variance
- Efficient memory management
- Automatic cleanup of resources

### Web Frontend

- A single‐page UI is served at the root (`GET /`), backed by `app/static/index.html`.
- All static assets are mounted under `/app/static`.
- CORS is enabled (via `CORSMiddleware`) so you can host the frontend and API on different origins.

## Technical Details

### API Endpoints

1. **Dataset Management**

   - `POST /datasets/` - Upload a new dataset
   - `GET /datasets/` - List all datasets
   - `GET /datasets/{id}/` - Get dataset information
   - `DELETE /datasets/{id}/` - Delete a dataset
   - `GET /datasets/{id}/excel/` - Export to Excel

2. **Analysis Endpoints**
   - `GET /datasets/{id}/stats/` - Get comprehensive dataset statistics
   - `GET /datasets/{id}/plot/` - Generate visualization PDF report

### Technical Stack

- **Backend Framework**: FastAPI
- **Data Processing**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **Server**: Uvicorn (ASGI)
- **File Handling**: Native Python I/O
- **Data Storage**: Local file system with JSON metadata

### Performance Considerations

- Maximum sample size for visualization: 10,000 rows
- Maximum columns to plot: 10 (selected by variance)
- Plots per page: 6 (2x3 grid)
- Automatic memory management
- Asynchronous processing for better performance

## Setup and Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app.main:app --reload
```

4. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

### Accessing the UI

1. After starting the server:

```bash
uvicorn app.main:app --reload
```

2. Open your browser at:

[http://localhost:8000/](http://localhost:8000/)

## Usage Examples

### Uploading a Dataset

```python
import requests

files = {'file': open('your_dataset.csv', 'rb')}
response = requests.post('http://localhost:8000/datasets/', files=files)
dataset_id = response.json()['id']
```

### Getting Visualizations

```python
response = requests.get(f'http://localhost:8000/datasets/{dataset_id}/plot/')
with open('visualizations.pdf', 'wb') as f:
    f.write(response.content)
```

### Exporting to Excel

```python
response = requests.get(f'http://localhost:8000/datasets/{dataset_id}/excel/')
with open('dataset.xlsx', 'wb') as f:
    f.write(response.content)
```

## Command Line Interface (CLI)

The project includes a command-line interface built with Click. Here are all available commands:

### List All Datasets

```bash
python client.py list
```

Lists all available datasets with their metadata (ID, filename, size, rows, and columns).

### Upload a Dataset

```bash
python client.py upload path/to/your/dataset.csv
```

Uploads a CSV file to the API. Only CSV files are supported.

### Get Dataset Information

```bash
python client.py info <dataset_id>
```

Displays detailed information about a specific dataset.

### Delete a Dataset

```bash
python client.py delete <dataset_id>
```

Deletes a dataset from the API.

### Export to Excel

```bash
python client.py export <dataset_id>
```

Exports the dataset to an Excel file.

### Get Dataset Statistics

```bash
python client.py stats <dataset_id>
```

Retrieves and displays comprehensive statistics for the dataset.

### Generate Visualizations

```bash
python client.py plot <dataset_id>
```

Generates and saves visualization plots for the dataset.

Note: Replace `<dataset_id>` with the actual dataset ID in all commands.

## Development

### Project Structure

```
dataset-analysis-api/
├── app/
│   ├── main.py
|   └── static/
|       └── index.html
├── datasets/
├── client.py
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
