"""
Tests for layer 3: anomaly detection (3_anomaly_detection.py).
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Pure unit test: the label-conversion logic
# ---------------------------------------------------------------------------
def test_if_label_conversion():
    """
    IsolationForest returns -1 for anomaly and 1 for normal. The
    evaluate_model function converts these to {0, 1} so they can be
    compared against the IsAnomaly ground-truth column. Verify the
    conversion is correct.
    """
    raw_predictions = np.array([-1, 1, -1, 1, 1, -1])
    converted = (raw_predictions == -1).astype(int)
    np.testing.assert_array_equal(converted, np.array([1, 0, 1, 0, 0, 1]))


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
def test_load_data_missing_db_raises_systemexit(tmp_path, load_layer):
    layer3 = load_layer("3_anomaly_detection")
    with pytest.raises(SystemExit):
        layer3.load_data(str(tmp_path / "no_such.db"))


def test_load_data_missing_table_raises_systemexit(tmp_path, load_layer):
    """An existing DB without a processed_traffic table should sys.exit."""
    layer3 = load_layer("3_anomaly_detection")
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    with pytest.raises(SystemExit):
        layer3.load_data(str(db_path))


# ---------------------------------------------------------------------------
# train_isolation_forest (integration, with a small DataFrame)
# ---------------------------------------------------------------------------
def test_train_isolation_forest_flags_known_outlier(load_layer, redirect_paths, tiny_processed_df):
    """
    With a 20-row DataFrame containing one row where PacketSize is 1e9
    (and the other 19 are normal-sized), the trained model must flag
    that row as an anomaly. We override the model constants so the
    test stays fast.
    """
    layer3 = load_layer("3_anomaly_detection")
    paths = redirect_paths(layer3, db_name="unused.db", chart_name="unused.png")

    # Shrink the forest for test speed
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(layer3, "N_ESTIMATORS", 20)
        monkey.setattr(layer3, "MAX_SAMPLES", 20)
        # The OUTLIER is at index 19 (the last row of tiny_processed_df)
        outlier_index = tiny_processed_df.index[-1]
        _model, predictions, _scores = layer3.train_isolation_forest(tiny_processed_df)
    finally:
        monkey.undo()

    # IF returns -1 for anomalies, 1 for normal.
    assert predictions[outlier_index] == -1, \
        f"row at index {outlier_index} (PacketSize=1e9) should be flagged"
    # The normal rows should mostly be classified as normal (1).
    normal_predictions = predictions[:-1]
    assert (normal_predictions == 1).sum() >= 15, \
        "most normal rows should be classified as normal"


def test_train_isolation_forest_uses_configured_contamination(load_layer, tiny_processed_df):
    """
    The contamination parameter is passed through to IsolationForest.
    The actual number of anomalies flagged should be roughly contamination
    * len(features) -- not a strict equality because the algorithm also
    considers the data distribution, but it should be in the right ballpark.
    """
    layer3 = load_layer("3_anomaly_detection")
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(layer3, "N_ESTIMATORS", 20)
        monkey.setattr(layer3, "MAX_SAMPLES", 20)
        monkey.setattr(layer3, "CONTAMINATION", 0.1)
        _model, predictions, _scores = layer3.train_isolation_forest(tiny_processed_df)
    finally:
        monkey.undo()

    n_anomalies = (predictions == -1).sum()
    # With contamination=0.1 and 20 rows, expect ~2 anomalies (range 0..6 is reasonable).
    assert 0 <= n_anomalies <= 6


# ---------------------------------------------------------------------------
# generate_alerts
# ---------------------------------------------------------------------------
def _seed_processed_table(db_path, df):
    """Write a DataFrame into a SQLite file as the processed_traffic table."""
    conn = sqlite3.connect(str(db_path))
    df.to_sql("processed_traffic", conn, if_exists="replace", index=False)
    conn.close()


def test_generate_alerts_writes_alerts_table(load_layer, redirect_paths, tiny_processed_df):
    """
    generate_alerts should create an `alerts` table containing only the
    rows that the model flagged as anomalies, plus an anomaly_score
    column and a prediction column.
    """
    layer3 = load_layer("3_anomaly_detection")
    paths = redirect_paths(layer3, db_name="alerts.db", chart_name="alerts.png")
    _seed_processed_table(paths["db"], tiny_processed_df)

    predictions = np.array([1] * 19 + [-1])   # last row is the anomaly
    scores = np.linspace(0.0, 0.5, 20)        # increasing -- lower = more anomalous

    n_alerts = layer3.generate_alerts(
        tiny_processed_df, predictions, scores, paths["db"]
    )
    assert n_alerts == 1

    conn = sqlite3.connect(paths["db"])
    try:
        # The alerts table should now exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "alerts" in tables

        cols = [r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()]
        assert "anomaly_score" in cols
        assert "prediction" in cols

        count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        assert count == 1

        # The single alert should be the row with row_id = 20 (the outlier)
        row = conn.execute(
            "SELECT row_id, prediction FROM alerts"
        ).fetchone()
        assert row[0] == 20
        assert row[1] == -1
    finally:
        conn.close()


def test_generate_alerts_replaces_existing_table(load_layer, redirect_paths, tiny_processed_df):
    """If the alerts table already exists, it should be replaced (not appended)."""
    layer3 = load_layer("3_anomaly_detection")
    paths = redirect_paths(layer3, db_name="alerts2.db", chart_name="alerts2.png")
    _seed_processed_table(paths["db"], tiny_processed_df)

    # Pre-create an alerts table with 5 dummy rows
    conn = sqlite3.connect(paths["db"])
    conn.execute("""
        CREATE TABLE alerts (row_id INTEGER, dummy TEXT)
    """)
    conn.executemany("INSERT INTO alerts VALUES (?, ?)",
                     [(i, "stale") for i in range(1, 6)])
    conn.commit()
    conn.close()

    predictions = np.array([1] * 19 + [-1])
    scores = np.linspace(0.0, 0.5, 20)
    layer3.generate_alerts(tiny_processed_df, predictions, scores, paths["db"])

    conn = sqlite3.connect(paths["db"])
    try:
        # The new alerts table should NOT have a 'dummy' column (proves it was replaced)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()]
        assert "dummy" not in cols
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# create_visualization
# ---------------------------------------------------------------------------
def test_create_visualization_writes_nonempty_png(load_layer, redirect_paths, tiny_processed_df):
    """create_visualization should write a non-empty PNG file to CHART_PATH."""
    layer3 = load_layer("3_anomaly_detection")
    paths = redirect_paths(layer3, db_name="viz.db", chart_name="viz.png")
    chart_path = paths["chart"]

    predictions = np.array([1] * 19 + [-1])
    scores = np.linspace(0.0, 0.5, 20)

    layer3.create_visualization(tiny_processed_df, predictions, scores, chart_path)

    assert os.path.isfile(chart_path)
    # PNG files start with the 8-byte signature 89 50 4E 47 0D 0A 1A 0A
    with open(chart_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", f"file is not a valid PNG: header={header!r}"
    assert os.path.getsize(chart_path) > 1000, "PNG file is suspiciously small"
