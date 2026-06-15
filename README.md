# Network Traffic Anomaly Detection Pipeline

A 3-layer network traffic anomaly detection pipeline that demonstrates **ETL**, **SQL-based feature engineering**, and **AI-powered anomaly detection** on a 1M-row synthetic network traffic dataset.

The end-to-end pipeline runs in **~67 seconds** and produces a SQLite database (`network_traffic.db`) and a scatter plot visualization (`anomaly_results.png`).

---

## Architecture

The pipeline is a strict 3-stage sequential flow. Each layer reads from / writes to the same `network_traffic.db` SQLite file — there is no inter-process communication beyond the database.

```
synthetic_network_traffic.csv
            │
            ▼
┌─────────────────────────────────────┐
│ Layer 1: ETL (1_data_ingestion.py)  │
│  • Chunked CSV → SQLite ingestion   │
│  • Synthesizes timestamps + row_id  │
│  • Writes raw_traffic table         │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────┐
│ Layer 2: SQL Transformations (2_sql_*.py)       │
│  1. Filter NULLs / |Protocol| > 3.0             │
│  2. 5-min window aggregation (per IP bin)       │
│  3. Heavy hitter flagging (mean + 2σ)           │
│  4. LEFT JOIN → processed_traffic (20 cols)     │
└─────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────┐
│ Layer 3: AI Detection (3_anomaly_detection.py)  │
│  • Isolation Forest on PacketSize + Duration    │
│  • Writes alerts table (anomaly_score)          │
│  • Renders anomaly_results.png                  │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- A virtual environment at `.venv/`

### Setup

```bash
# Create and activate the virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python run_pipeline.py
```

The runner imports each layer in order, prints a timing summary, and exits with code `1` plus a stack trace if any layer fails.

### Run individual layers (in order)

```bash
python 1_data_ingestion.py        # Layer 1: CSV → SQLite ETL
python 2_sql_transformations.py   # Layer 2: SQL feature engineering
python 3_anomaly_detection.py     # Layer 3: Isolation Forest detection
```

> **Note:** Running layers out of order will fail with a clear error message — each layer assumes the previous layer's output table exists.

---

## Project Structure

```
project/
├── 1_data_ingestion.py          # Layer 1: CSV → SQLite ETL
├── 2_sql_transformations.py     # Layer 2: SQL feature engineering
├── 3_anomaly_detection.py       # Layer 3: Isolation Forest detection
├── run_pipeline.py              # Master pipeline runner
├── requirements.txt             # Pinned Python dependencies
├── synthetic_network_traffic.csv  # Source dataset (1M rows, ~190MB)
├── network_traffic.db           # Generated SQLite database
├── anomaly_results.png          # Generated scatter plot
├── tests/                       # pytest test suite
│   ├── conftest.py
│   ├── test_1_data_ingestion.py
│   ├── test_2_sql_transformations.py
│   ├── test_3_anomaly_detection.py
│   └── test_pipeline_runner.py
└── .venv/                       # Virtual environment (not in git)
```

---

## Pipeline Details

### Layer 1 — ETL (`1_data_ingestion.py`)

- Reads `synthetic_network_traffic.csv` in **50K-row chunks** via `pandas.read_csv(chunksize=...)` to keep peak memory around **60MB**.
- Generates synthetic timestamps (1 second apart, starting `2024-01-01 00:00:00`) and a sequential `row_id` primary key.
- Writes everything to a fresh `raw_traffic` table (overwrites any existing DB at startup).
- Creates indexes on `timestamp`, `SourceIP`, `IsAnomaly` for downstream query speed.

### Layer 2 — SQL Transformations (`2_sql_transformations.py`)

Operates entirely in SQL (no pandas DataFrame work) against the `raw_traffic` table.

- **Step 1 (filter):** deletes rows with NULLs in key columns or `|Protocol| > 3.0` (extreme outliers).
- **Step 2 (aggregate):** 5-minute window aggregation grouped by `SourceIP` rounded to 0.1 (since `SourceIP` values are z-score floats, not real IPs). Windows are computed via `strftime` minute-flooring. Produces `ip_window_aggregates`.
- **Step 3 (heavy hitters):** SQLite has no `STDEV`, so mean and stdev of `total_bytes_sent` are computed in Python (`flag_heavy_hitters` pulls all values into a list), then injected as a parameter into a parameterized `CREATE TABLE … WHERE total_bytes_sent > ?` query. Threshold is `mean + 2 * stdev`.
- **Step 4 (join):** builds the final `processed_traffic` table by `LEFT JOIN`-ing cleaned `raw_traffic` with `ip_window_aggregates` and `heavy_hitters` on `(source_ip_bin, window_start)`. Adds 7 derived columns: `PacketSize`, `source_ip_bin`, `window_total_bytes`, `window_conn_count`, `window_avg_duration`, `window_anomaly_count`, `is_heavy_hitter`.

### Layer 3 — AI Detection (`3_anomaly_detection.py`)

- Loads `processed_traffic` into a pandas DataFrame (full 1M rows, ~997K after filtering).
- Trains an `IsolationForest` (`contamination=0.05`, `n_estimators=100`, `max_samples=50000`, `random_state=42`, `n_jobs=-1`) on only **two features**: `PacketSize` and `Duration`.
- Evaluates against the ground-truth `IsAnomaly` column (precision, recall, F1, confusion matrix). The low precision/recall is expected — the synthetic anomalies are not strongly correlated with just these two features.
- Writes predicted anomalies to a new `alerts` table (replaces if exists), sorted most-anomalous first by `anomaly_score`.
- Produces `anomaly_results.png` — a 2-panel scatter plot (predictions + anomaly-score heatmap) using a 20K-point subsample. Uses `matplotlib.use("Agg")` for headless rendering.

### Pipeline runner (`run_pipeline.py`)

- Imports each layer module via `importlib.import_module("1_data_ingestion")` etc. (note the numeric prefix — Python allows this but linters may flag it). Each layer exposes a `run()` function returning a row count.
- Exits with code 1 and a stack trace if any layer fails.

---

## Database Tables (post-pipeline)

| Table                    | Rows     | Notes                                  |
|--------------------------|----------|----------------------------------------|
| `raw_traffic`            | 997,298  | After filter step                      |
| `ip_window_aggregates`   | 161,734  | Per IP-bin × 5-min window              |
| `heavy_hitters`          | 4,793    | Above `mean + 2σ`                      |
| `processed_traffic`      | 997,298  | Joined feature table for ML            |
| `alerts`                 | 49,865   | Isolation Forest output                |

---

## Running the Tests

```bash
pytest tests/
```

The test suite uses `tests/conftest.py` to share fixtures and a temporary SQLite database so tests do not clobber `network_traffic.db`.

---

## Dependencies

Pinned in `requirements.txt`:

| Package       | Version  |
|---------------|----------|
| pandas        | 2.2.3    |
| scikit-learn  | 1.6.1    |
| matplotlib    | 3.10.0   |
| seaborn       | 0.13.2   |
| tabulate      | 0.9.0    |
| numpy         | 2.2.3    |
| pytest        | 8.3.4    |

---

## Key Conventions & Gotchas

- **`SourceIP` / `DestinationIP` are z-score floats, not real IPs.** They are binned via `ROUND(SourceIP, 1)` for grouping. Do not try to parse them as strings.
- **SQLite has no `STDEV()`.** The heavy-hitter threshold is computed in Python by loading all aggregate values into memory. Fine at 161K rows; would not scale.
- **Each layer's `run()` overwrites the database / table it owns** when re-run individually:
  - `1_data_ingestion.py` deletes `network_traffic.db` entirely.
  - `2_sql_transformations.py` drops `ip_window_aggregates`, `heavy_hitters`, and `processed_traffic` before recreating.
  - `3_anomaly_detection.py` uses `to_sql(if_exists="replace")` for `alerts`.
- **Layer 3 only uses 2 of the 20 features.** This is intentional for the demo but limits model quality. To improve, expand the `features` DataFrame in `train_isolation_forest` to include all 20 columns.
