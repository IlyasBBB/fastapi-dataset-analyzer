# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- A recruiter or engineer reviewing Ilyas Ben Alla's work: opens the page, uploads a CSV (or a provided sample), and should see in under a minute what the API does.
- Origin: a technical test for an internship application (May 2025). The brief asked for the REST API and a command-line client; the web page was added on top of the brief.

## Product Purpose

A FastAPI service that stores uploaded CSV datasets, returns their metadata and descriptive statistics, exports them to Excel and generates a multi-page PDF report of charts (histograms with KDE, box plots, value-vs-index lines, scatterplot matrix, correlation heatmap).

## Capabilities and Constraints

- Endpoints: POST/GET `/datasets/`, GET/DELETE `/datasets/{id}/`, `/excel/`, `/stats/`, `/plot/`. Interactive docs at `/docs`.
- CSV only, 20 MB limit, charts on a 10,000-row sample and the 10 highest-variance numeric columns.
- The web page is a single static file served by FastAPI (`app/static/index.html`), plain JavaScript, no build step.
- Interface and documentation in English (British spelling).

## Evidence on Hand

- The code and its tests. No usage figures, no users: none may be invented.

## Product Principles

1. Every number on screen comes from the API response, never from the page.
2. The page shows the API at work: requests visible, responses readable.
3. A first-time visitor can try it with one click.
