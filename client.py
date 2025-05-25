import click
import requests
import os
from pathlib import Path
import json

API_URL = "http://localhost:8000"

@click.group()
def cli():
    """Dataset API Client"""
    pass

@cli.command()
def list():
    """List all datasets"""
    response = requests.get(f"{API_URL}/datasets/")
    if response.status_code == 200:
        datasets = response.json()
        if not datasets:
            click.echo("No datasets found")
            return
        
        for dataset in datasets:
            click.echo(f"ID: {dataset['id']}")
            click.echo(f"Filename: {dataset['filename']}")
            click.echo(f"Size: {dataset['size']} bytes")
            click.echo(f"Rows: {dataset['rows']}")
            click.echo(f"Columns: {dataset['columns']}")
            click.echo("---")
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def upload(file_path):
    """Upload a CSV file"""
    if not file_path.endswith('.csv'):
        click.echo("Error: Only CSV files are supported")
        return
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/csv')}
        response = requests.post(f"{API_URL}/datasets/", files=files)
    
    if response.status_code == 200:
        dataset = response.json()
        click.echo(f"Dataset uploaded successfully!")
        click.echo(f"ID: {dataset['id']}")
        click.echo(f"Filename: {dataset['filename']}")
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('dataset_id')
def info(dataset_id):
    """Get dataset information"""
    response = requests.get(f"{API_URL}/datasets/{dataset_id}/")
    if response.status_code == 200:
        dataset = response.json()
        click.echo(f"ID: {dataset['id']}")
        click.echo(f"Filename: {dataset['filename']}")
        click.echo(f"Size: {dataset['size']} bytes")
        click.echo(f"Rows: {dataset['rows']}")
        click.echo(f"Columns: {dataset['columns']}")
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('dataset_id')
def delete(dataset_id):
    """Delete a dataset"""
    response = requests.delete(f"{API_URL}/datasets/{dataset_id}/")
    if response.status_code == 200:
        click.echo("Dataset deleted successfully")
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('dataset_id')
def export(dataset_id):
    """Export dataset to Excel"""
    response = requests.get(f"{API_URL}/datasets/{dataset_id}/excel/")
    if response.status_code == 200:
        filename = response.headers.get('content-disposition', '').split('filename=')[-1].strip('"')
        with open(filename, 'wb') as f:
            f.write(response.content)
        click.echo(f"Dataset exported to {filename}")
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('dataset_id')
def stats(dataset_id):
    """Get dataset statistics"""
    response = requests.get(f"{API_URL}/datasets/{dataset_id}/stats/")
    if response.status_code == 200:
        stats = response.json()
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo(f"Error: {response.text}")

@cli.command()
@click.argument('dataset_id')
def plot(dataset_id):
    """Generate plots for numerical columns"""
    response = requests.get(f"{API_URL}/datasets/{dataset_id}/plot/")
    if response.status_code == 200:
        filename = response.headers.get('content-disposition', '').split('filename=')[-1].strip('"')
        with open(filename, 'wb') as f:
            f.write(response.content)
        click.echo(f"Plots saved to {filename}")
    else:
        click.echo(f"Error: {response.text}")

if __name__ == '__main__':
    cli() 