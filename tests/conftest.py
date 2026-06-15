"""
Shared pytest fixtures for the anomaly-detection pipeline tests.

The layer scripts use module-level path constants (CSV_PATH, DB_PATH,
CHART_PATH). To test them against a tiny synthetic CSV and a fresh
database, tests monkeypatch those constants to point inside pytest's
per-test tmp_path. The conftest also injects the project root into
sys.path so the numeric-prefix module names (1_data_ingestion, ...)
import cleanly.
"""

import os
import sys
import csv

import pytest


# ---------------------------------------------------------------------------
# Path setup: make the project root importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------
# Schema mirrors the columns produced by pandas.read_csv on the real
# synthetic_network_traffic.csv. Layer 1 expects these 10 columns in this
# order; it adds row_id and timestamp itself.
CSV_COLUMNS = [
    "SourceIP",
    "DestinationIP",
    "SourcePort",
    "DestinationPort",
    "Protocol",
    "BytesSent",
    "BytesReceived",
    "PacketsSent",
    "PacketsReceived",
    "Duration",
    "IsAnomaly",
]


def _make_csv(path, rows):
    """Write a list of row-dicts to `path` using CSV_COLUMNS ordering."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _good_row(seed):
    """A realistic-looking non-anomalous row."""
    return {
        "SourceIP":         round(0.1 * (seed % 100), 3),
        "DestinationIP":    round(0.1 * ((seed * 7) % 100), 3),
        "SourcePort":       1000 + (seed % 50000),
        "DestinationPort":  80 if seed % 3 else 443,
        "Protocol":         round(((seed * 13) % 5) - 2.0, 4),  # in [-2, 2]
        "BytesSent":        500 + (seed * 37) % 10000,
        "BytesReceived":    1000 + (seed * 53) % 20000,
        "PacketsSent":      1 + (seed * 7) % 50,
        "PacketsReceived":  1 + (seed * 11) % 80,
        "Duration":         round(0.1 + (seed % 100) * 0.05, 4),
        "IsAnomaly":        0,
    }


def _anomaly_row(seed):
    """A row flagged as a true anomaly in the ground truth."""
    row = _good_row(seed)
    row["IsAnomaly"] = 1
    row["BytesSent"] = 9_000_000 + seed
    return row


@pytest.fixture
def tiny_csv(tmp_path):
    """
    Write a 30-row synthetic CSV to tmp_path / "tiny_traffic.csv".

    Composition:
      - 27 normal rows
      - 1 known-anomaly row (IsAnomaly=1)
      - 1 row with NULL BytesSent  (triggers the filter step)
      - 1 row with Protocol = 5.0  (triggers the |Protocol| > 3.0 filter)

    Returned path contains all 30 rows pre-filter; layer 2 should drop 2.
    """
    path = tmp_path / "tiny_traffic.csv"
    rows = [_good_row(i) for i in range(27)]
    rows.append(_anomaly_row(100))
    # NULL row -- csv.DictWriter writes "" for None, which pandas reads as NaN
    null_row = _good_row(200)
    null_row["BytesSent"] = None
    rows.append(null_row)
    # Protocol outlier
    out_row = _good_row(300)
    out_row["Protocol"] = 5.0
    rows.append(out_row)
    _make_csv(path, rows)
    return path


@pytest.fixture
def clean_csv(tmp_path):
    """
    A 25-row CSV with no NULLs, no outliers, no anomalies.
    Useful for tests that only need well-formed input.
    """
    path = tmp_path / "clean_traffic.csv"
    rows = [_good_row(i) for i in range(25)]
    _make_csv(path, rows)
    return path


@pytest.fixture
def tiny_processed_df():
    """
    A small DataFrame shaped like processed_traffic for layer 3 unit tests.
    Contains one obvious outlier (PacketSize = 1e9) and 19 normal rows.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    normal_size = rng.normal(loc=2000, scale=500, size=19)
    normal_dur = rng.normal(loc=5.0, scale=1.0, size=19)
    df = pd.DataFrame({
        "row_id":          range(1, 21),
        "timestamp":       ["2024-01-01 00:00:00"] * 20,
        "SourceIP":        [0.1] * 20,
        "DestinationIP":   [0.2] * 20,
        "PacketSize":      np.concatenate([normal_size, [1e9]]),
        "Duration":        np.concatenate([normal_dur, [5.0]]),
        "BytesSent":       [1000] * 20,
        "BytesReceived":   [1000] * 20,
        "Protocol":        [1.0] * 20,
        "IsAnomaly":       [0] * 19 + [1],
        "is_heavy_hitter": [0] * 20,
    })
    return df


# ---------------------------------------------------------------------------
# Path-redirect fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def redirect_paths(tmp_path, monkeypatch):
    """
    Return a helper that patches CSV_PATH, DB_PATH, and CHART_PATH on a
    given module to point inside tmp_path. Attributes that don't exist
    on the module (e.g. CHART_PATH on layer 1) are silently skipped.

    Usage:
        redirect_paths(layer1, csv_path=tiny_csv, db_name="layer1.db")
    """
    def _apply(module, *, csv_path=None, db_name="test.db", chart_name="chart.png"):
        db_path = str(tmp_path / db_name)
        chart_path = str(tmp_path / chart_name)
        if csv_path is not None and hasattr(module, "CSV_PATH"):
            monkeypatch.setattr(module, "CSV_PATH", str(csv_path))
        if hasattr(module, "DB_PATH"):
            monkeypatch.setattr(module, "DB_PATH", db_path)
        if hasattr(module, "CHART_PATH"):
            monkeypatch.setattr(module, "CHART_PATH", chart_path)
        return {"db": db_path, "chart": chart_path}

    return _apply


# ---------------------------------------------------------------------------
# Convenience: load a layer module by its numeric-prefix filename
# ---------------------------------------------------------------------------
@pytest.fixture
def load_layer():
    """
    Returns a function: name -> module.

    Uses importlib so the numeric-prefix module names (1_data_ingestion,
    2_sql_transformations, 3_anomaly_detection) load correctly.
    """
    import importlib

    def _load(name):
        return importlib.import_module(name)

    return _load
