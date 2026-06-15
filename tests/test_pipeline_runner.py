"""
End-to-end test for run_pipeline.main.

This is the canary test that catches wiring regressions: it runs the
full 3-layer pipeline against a 30-row synthetic CSV in tmp_path and
asserts that the resulting database and PNG are produced.
"""

import importlib
import os
import sqlite3
import sys

import pytest


def test_run_pipeline_end_to_end_on_tiny_csv(tmp_path, monkeypatch, tiny_csv):
    """
    Run run_pipeline.main with all module-level paths redirected to
    tmp_path, then verify the resulting artifacts.
    """
    # Make tmp_path the working directory so relative paths in the
    # production scripts (e.g. 'network_traffic.db') don't accidentally
    # collide with the real files in the project root.
    monkeypatch.chdir(tmp_path)

    # Load every layer and override its path constants to use tmp_path.
    # Layer 1 uses CSV_PATH and DB_PATH.
    layer1 = importlib.import_module("1_data_ingestion")
    monkeypatch.setattr(layer1, "CSV_PATH", str(tiny_csv))
    monkeypatch.setattr(layer1, "DB_PATH", str(tmp_path / "pipeline.db"))

    # Layer 2 only uses DB_PATH.
    layer2 = importlib.import_module("2_sql_transformations")
    monkeypatch.setattr(layer2, "DB_PATH", str(tmp_path / "pipeline.db"))

    # Layer 3 uses DB_PATH and CHART_PATH.
    layer3 = importlib.import_module("3_anomaly_detection")
    monkeypatch.setattr(layer3, "DB_PATH", str(tmp_path / "pipeline.db"))
    monkeypatch.setattr(layer3, "CHART_PATH", str(tmp_path / "pipeline.png"))

    # Shrink layer 3 for test speed
    monkeypatch.setattr(layer3, "N_ESTIMATORS", 20)
    monkeypatch.setattr(layer3, "MAX_SAMPLES", 30)

    # Now reload run_pipeline (it does its own sys.path manipulation in
    # module-level code -- but the path is already set by conftest).
    if "run_pipeline" in sys.modules:
        del sys.modules["run_pipeline"]
    run_pipeline = importlib.import_module("run_pipeline")

    # Suppress the printed banner / progress output so test logs stay clean.
    monkeypatch.setattr(run_pipeline, "print_banner", lambda: None)
    monkeypatch.setattr(run_pipeline, "print_summary", lambda *a, **k: None)

    # The pipeline mutates sys.path on import; restore it to what it was
    # before the test started so we don't pollute global state.
    original_path = list(sys.path)
    try:
        run_pipeline.main()
    finally:
        # run_pipeline.main appends to sys.path; remove any new entries
        for entry in sys.path:
            if entry not in original_path:
                sys.path.remove(entry)

    # --- Assertions on the resulting artifacts ---

    # 1. The database was created in tmp_path
    db_path = tmp_path / "pipeline.db"
    assert db_path.exists(), "pipeline.db was not created in tmp_path"

    # 2. The database has all expected tables
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    finally:
        conn.close()

    expected_tables = {
        "raw_traffic", "ip_window_aggregates", "heavy_hitters",
        "processed_traffic", "alerts",
    }
    assert expected_tables.issubset(tables), \
        f"missing tables: {expected_tables - tables}"

    # 3. The alerts table is non-empty (some rows were flagged in the
    #    30-row seed -- even if the model is conservative, the pipeline
    #    ran to completion).
    conn = sqlite3.connect(str(db_path))
    try:
        n_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    finally:
        conn.close()
    assert n_alerts >= 1, "alerts table is empty after pipeline run"

    # 4. The PNG visualization was created and is a valid PNG
    chart_path = tmp_path / "pipeline.png"
    assert chart_path.exists(), "pipeline.png was not created in tmp_path"
    assert os.path.getsize(chart_path) > 1000, "pipeline.png is suspiciously small"
    with open(chart_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "pipeline.png is not a valid PNG"


def test_run_pipeline_propagates_layer_validation_failure(tmp_path, monkeypatch, tiny_csv, capsys):
    """
    If a layer's validation step fails (e.g. validate_csv can't find the
    CSV), it raises SystemExit(1) directly. run_pipeline.main propagates
    that exit -- this is the contract the pipeline runner currently
    implements. The test pins that behavior.
    """
    # Force layer 1 to point at a non-existent CSV
    layer1 = importlib.import_module("1_data_ingestion")
    monkeypatch.setattr(layer1, "CSV_PATH", str(tmp_path / "does_not_exist.csv"))
    monkeypatch.setattr(layer1, "DB_PATH", str(tmp_path / "fail.db"))

    if "run_pipeline" in sys.modules:
        del sys.modules["run_pipeline"]
    run_pipeline = importlib.import_module("run_pipeline")
    monkeypatch.setattr(run_pipeline, "print_banner", lambda: None)
    monkeypatch.setattr(run_pipeline, "print_summary", lambda *a, **k: None)

    original_path = list(sys.path)
    try:
        with pytest.raises(SystemExit) as exc_info:
            run_pipeline.main()
    finally:
        for entry in sys.path:
            if entry not in original_path:
                sys.path.remove(entry)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "CSV file not found" in out
