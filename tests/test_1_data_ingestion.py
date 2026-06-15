"""
Tests for layer 1: data ingestion (1_data_ingestion.py).
"""

import os
import sqlite3

import pandas as pd
import pytest

from tests.conftest import CSV_COLUMNS


# ---------------------------------------------------------------------------
# validate_csv
# ---------------------------------------------------------------------------
def test_validate_csv_missing_file_raises_systemexit(tmp_path, capsys, load_layer):
    """validate_csv should sys.exit(1) when the CSV does not exist."""
    layer1 = load_layer("1_data_ingestion")
    missing = tmp_path / "nope.csv"
    with pytest.raises(SystemExit) as exc_info:
        layer1.validate_csv(str(missing))
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "CSV file not found" in captured.out


def test_validate_csv_existing_file_passes(load_layer, tiny_csv, capsys):
    """validate_csv should not exit when the file exists; prints size."""
    layer1 = load_layer("1_data_ingestion")
    layer1.validate_csv(str(tiny_csv))  # should not raise
    captured = capsys.readouterr()
    assert "Source CSV found" in captured.out
    assert "MB" in captured.out


# ---------------------------------------------------------------------------
# create_database
# ---------------------------------------------------------------------------
def test_create_database_replaces_existing(tmp_path, load_layer):
    """If a DB already exists at db_path, its contents should be removed."""
    layer1 = load_layer("1_data_ingestion")
    db_path = tmp_path / "test.db"

    # Pre-create a dummy file with sentinel content
    db_path.write_text("placeholder")
    assert db_path.exists()

    conn = layer1.create_database(str(db_path))
    try:
        # Should be a real SQLite connection now, not a text file
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()

    # The placeholder content should be gone -- SQLite replaced it with
    # an empty file (header is only written on first commit).
    assert db_path.exists()
    assert db_path.read_text() == "", "placeholder content should have been removed"


def test_create_database_returns_connection(tmp_path, load_layer):
    """create_database should return a working sqlite3 connection."""
    layer1 = load_layer("1_data_ingestion")
    db_path = tmp_path / "fresh.db"
    conn = layer1.create_database(str(db_path))
    try:
        result = conn.execute("SELECT 42").fetchone()[0]
        assert result == 42
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# create_table
# ---------------------------------------------------------------------------
def test_create_table_has_expected_columns(tmp_path, load_layer):
    """raw_traffic should have 12 columns with the expected names and types."""
    layer1 = load_layer("1_data_ingestion")
    db_path = tmp_path / "schema.db"
    conn = layer1.create_database(str(db_path))
    try:
        layer1.create_table(conn)
        cols = conn.execute("PRAGMA table_info(raw_traffic)").fetchall()
        col_names = [c[1] for c in cols]
        col_types = {c[1]: c[2] for c in cols}
    finally:
        conn.close()

    expected = [
        "row_id", "timestamp", "SourceIP", "DestinationIP", "SourcePort",
        "DestinationPort", "Protocol", "BytesSent", "BytesReceived",
        "PacketsSent", "PacketsReceived", "Duration", "IsAnomaly",
    ]
    assert col_names == expected
    assert col_types["row_id"] == "INTEGER"
    assert col_types["timestamp"] == "TEXT"
    assert col_types["IsAnomaly"] == "INTEGER"


def test_create_table_is_idempotent_when_called_twice(tmp_path, load_layer):
    """create_table drops the table first, so a second call should not error."""
    layer1 = load_layer("1_data_ingestion")
    db_path = tmp_path / "twice.db"
    conn = layer1.create_database(str(db_path))
    try:
        layer1.create_table(conn)
        layer1.create_table(conn)  # would fail without the DROP IF EXISTS
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ingest_csv (integration)
# ---------------------------------------------------------------------------
def test_ingest_csv_row_count_and_row_ids(load_layer, redirect_paths, tiny_csv):
    """ingest_csv should write every CSV row, with sequential 1-based row_ids."""
    layer1 = load_layer("1_data_ingestion")
    paths = redirect_paths(layer1, csv_path=tiny_csv, db_name="ingest.db")
    conn = layer1.create_database(paths["db"])
    try:
        layer1.create_table(conn)
        total = layer1.ingest_csv(str(tiny_csv), conn)
    finally:
        conn.close()

    # tiny_csv has 30 rows total (27 normal + 1 anomaly + 1 null + 1 outlier)
    assert total == 30

    # Re-open and inspect
    conn = sqlite3.connect(paths["db"])
    try:
        df = pd.read_sql_query("SELECT * FROM raw_traffic ORDER BY row_id", conn)
    finally:
        conn.close()

    assert len(df) == 30
    assert list(df["row_id"]) == list(range(1, 31))
    # Every source CSV column should be present
    for col in CSV_COLUMNS:
        assert col in df.columns


def test_ingest_csv_timestamps_are_sequential_seconds(load_layer, redirect_paths, tiny_csv):
    """Timestamps should start at START_TIMESTAMP and step 1 second apart."""
    from datetime import datetime, timedelta

    layer1 = load_layer("1_data_ingestion")
    paths = redirect_paths(layer1, csv_path=tiny_csv, db_name="ts.db")
    conn = layer1.create_database(paths["db"])
    try:
        layer1.create_table(conn)
        layer1.ingest_csv(str(tiny_csv), conn)
    finally:
        conn.close()

    conn = sqlite3.connect(paths["db"])
    try:
        rows = conn.execute(
            "SELECT row_id, timestamp FROM raw_traffic ORDER BY row_id"
        ).fetchall()
    finally:
        conn.close()

    # The first row's timestamp should equal START_TIMESTAMP formatted as string
    expected_first = layer1.START_TIMESTAMP.strftime("%Y-%m-%d %H:%M:%S")
    assert rows[0][1] == expected_first

    # Every subsequent row's timestamp should be exactly +1 second from the prior
    for prev, curr in zip(rows, rows[1:]):
        prev_ts = datetime.strptime(prev[1], "%Y-%m-%d %H:%M:%S")
        curr_ts = datetime.strptime(curr[1], "%Y-%m-%d %H:%M:%S")
        assert curr_ts - prev_ts == timedelta(seconds=1)
        # And the row_id delta must match the timestamp delta
        assert curr[0] - prev[0] == 1


def test_ingest_csv_creates_indexes_when_run_end_to_end(load_layer, redirect_paths, tiny_csv):
    """The full run() should leave indexes on timestamp, SourceIP, IsAnomaly."""
    layer1 = load_layer("1_data_ingestion")
    paths = redirect_paths(layer1, csv_path=tiny_csv, db_name="idx.db")
    layer1.run()
    # run() created its own DB using the (now-monkeypatched) DB_PATH

    conn = sqlite3.connect(paths["db"])
    try:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='raw_traffic'"
        ).fetchall()
    finally:
        conn.close()

    index_names = {row[0] for row in indexes}
    # SQLite creates an implicit index for the PRIMARY KEY; the script's
    # named indexes should also be present.
    for required in ("idx_timestamp", "idx_source_ip", "idx_anomaly"):
        assert required in index_names, f"missing index: {required}"


# ---------------------------------------------------------------------------
# verify_ingestion
# ---------------------------------------------------------------------------
def test_verify_ingestion_prints_row_count_and_anomaly_split(
    load_layer, redirect_paths, tiny_csv, capsys
):
    """verify_ingestion should report the total row count and the
    IsAnomaly distribution (1 anomaly of 30 in the tiny CSV)."""
    layer1 = load_layer("1_data_ingestion")
    paths = redirect_paths(layer1, csv_path=tiny_csv, db_name="verify.db")
    conn = layer1.create_database(paths["db"])
    try:
        layer1.create_table(conn)
        layer1.ingest_csv(str(tiny_csv), conn)
        layer1.verify_ingestion(conn)
    finally:
        conn.close()

    out = capsys.readouterr().out
    assert "Total rows loaded" in out
    assert "30" in out
    # The one anomaly row should appear with a count of 1
    assert "IsAnomaly=1" in out


# ---------------------------------------------------------------------------
# run() smoke test
# ---------------------------------------------------------------------------
def test_run_returns_total_row_count(load_layer, redirect_paths, tiny_csv):
    """run() should return the total number of rows ingested."""
    layer1 = load_layer("1_data_ingestion")
    redirect_paths(layer1, csv_path=tiny_csv, db_name="run.db")
    total = layer1.run()
    assert total == 30
