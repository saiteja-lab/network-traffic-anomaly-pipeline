"""
Tests for layer 2: SQL transformations (2_sql_transformations.py).
"""

import math
import sqlite3

import pandas as pd
import pytest

from tests.conftest import CSV_COLUMNS


# ---------------------------------------------------------------------------
# Helper: build a fully-prepared DB for layer-2 tests.
# Layer 2 expects raw_traffic to already exist (produced by layer 1).
# ---------------------------------------------------------------------------
def _seed_raw_traffic(db_path, n_normal=20, n_null=0, n_outlier=0, n_anomaly=0):
    """
    Insert a synthetic raw_traffic table at db_path with the schema that
    1_data_ingestion.create_table would have produced.

    Returns the sqlite3 connection (caller is responsible for closing).
    """
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS raw_traffic;
        CREATE TABLE raw_traffic (
            row_id          INTEGER PRIMARY KEY,
            timestamp       TEXT    NOT NULL,
            SourceIP        REAL,
            DestinationIP   REAL,
            SourcePort      REAL,
            DestinationPort REAL,
            Protocol        REAL,
            BytesSent       REAL,
            BytesReceived   REAL,
            PacketsSent     REAL,
            PacketsReceived REAL,
            Duration        REAL,
            IsAnomaly       INTEGER
        );
    """)
    rows = []
    next_id = 1

    for i in range(n_normal):
        rows.append((
            next_id, f"2024-01-01 00:00:{i:02d}",
            0.1 * (i % 10), 0.2,
            1000 + i, 80,
            0.5, 1000.0, 2000.0, 5, 10, 1.0, 0,
        ))
        next_id += 1

    for i in range(n_null):
        rows.append((
            next_id, f"2024-01-01 00:01:{i:02d}",
            0.1, 0.2, 1000, 80, 0.5,
            None,    # NULL BytesSent -- triggers the filter
            2000.0, 5, 10, 1.0, 0,
        ))
        next_id += 1

    for i in range(n_outlier):
        rows.append((
            next_id, f"2024-01-01 00:02:{i:02d}",
            0.1, 0.2, 1000, 80,
            5.0,    # |Protocol| > 3.0 -- triggers the filter
            1000.0, 2000.0, 5, 10, 1.0, 0,
        ))
        next_id += 1

    for i in range(n_anomaly):
        rows.append((
            next_id, f"2024-01-01 00:03:{i:02d}",
            0.1, 0.2, 1000, 80, 0.5,
            9_000_000.0, 2000.0, 5, 10, 1.0, 1,
        ))
        next_id += 1

    conn.executemany(
        """INSERT INTO raw_traffic
           (row_id, timestamp, SourceIP, DestinationIP, SourcePort, DestinationPort,
            Protocol, BytesSent, BytesReceived, PacketsSent, PacketsReceived,
            Duration, IsAnomaly)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# connect_db
# ---------------------------------------------------------------------------
def test_connect_db_missing_file_raises_systemexit(tmp_path, load_layer):
    layer2 = load_layer("2_sql_transformations")
    with pytest.raises(SystemExit):
        layer2.connect_db(str(tmp_path / "no_such.db"))


def test_connect_db_missing_table_raises_systemexit(tmp_path, load_layer):
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    with pytest.raises(SystemExit):
        layer2.connect_db(str(db_path))


def test_connect_db_returns_connection(tmp_path, load_layer):
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "ok.db"
    _seed_raw_traffic(db_path, n_normal=3).close()
    conn = layer2.connect_db(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM raw_traffic").fetchone()[0]
        assert count == 3
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# filter_invalid_data
# ---------------------------------------------------------------------------
def test_filter_invalid_data_removes_nulls_and_outliers(load_layer, tmp_path):
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "filter.db"
    conn = _seed_raw_traffic(
        db_path, n_normal=20, n_null=2, n_outlier=1, n_anomaly=1
    )
    try:
        removed = layer2.filter_invalid_data(conn)
    finally:
        conn.close()

    # 2 NULLs + 1 protocol outlier = 3 rows removed
    assert removed == 3

    # Reopen and verify
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM raw_traffic").fetchone()[0]
        # 20 + 2 + 1 + 1 = 24 starting rows - 3 removed = 21
        assert count == 21
        # No NULL BytesSent should remain
        nulls = conn.execute(
            "SELECT COUNT(*) FROM raw_traffic WHERE BytesSent IS NULL"
        ).fetchone()[0]
        assert nulls == 0
        # No |Protocol| > 3.0 should remain
        outliers = conn.execute(
            "SELECT COUNT(*) FROM raw_traffic WHERE ABS(Protocol) > 3.0"
        ).fetchone()[0]
        assert outliers == 0
    finally:
        conn.close()


def test_filter_invalid_data_no_op_on_clean_data(load_layer, tmp_path):
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "clean.db"
    conn = _seed_raw_traffic(db_path, n_normal=10)
    try:
        removed = layer2.filter_invalid_data(conn)
    finally:
        conn.close()
    assert removed == 0


# ---------------------------------------------------------------------------
# aggregate_by_window
# ---------------------------------------------------------------------------
def test_aggregate_by_window_groups_by_ip_bin_and_5min(load_layer, tmp_path):
    """
    All 20 normal rows use SourceIP = 0.1..1.0 (mod 10) and timestamps
    in the first minute of 2024-01-01, so they all fall into a single
    5-minute window. We expect 10 (ip_bin, window) groups -- one per
    distinct SourceIP.
    """
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "agg.db"
    conn = _seed_raw_traffic(db_path, n_normal=20)
    try:
        agg_rows = layer2.aggregate_by_window(conn)
    finally:
        conn.close()

    # 20 rows / 10 distinct IPs (0.1 through 1.0 in 0.1 steps) = 10 groups
    assert agg_rows == 10

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT source_ip_bin, window_start, connection_count "
            "FROM ip_window_aggregates ORDER BY source_ip_bin",
            conn,
        )
    finally:
        conn.close()

    # Every window_start should equal the floored-to-5-min value
    # 00:00:00 floored to minute 0 = 00:00:00
    assert (df["window_start"] == "2024-01-01 00:00:00").all()
    # Each IP bin should have exactly 2 connections
    assert (df["connection_count"] == 2).all()


def test_aggregate_by_window_creates_different_windows(load_layer, tmp_path):
    """
    Two rows in different 5-minute buckets should produce two groups
    even when they share the same SourceIP.
    """
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "agg2.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        DROP TABLE IF EXISTS raw_traffic;
        CREATE TABLE raw_traffic (
            row_id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            SourceIP REAL, DestinationIP REAL, SourcePort REAL,
            DestinationPort REAL, Protocol REAL, BytesSent REAL,
            BytesReceived REAL, PacketsSent REAL, PacketsReceived REAL,
            Duration REAL, IsAnomaly INTEGER
        );
    """)
    conn.executemany(
        """INSERT INTO raw_traffic
           (row_id, timestamp, SourceIP, DestinationIP, SourcePort, DestinationPort,
            Protocol, BytesSent, BytesReceived, PacketsSent, PacketsReceived,
            Duration, IsAnomaly)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (1, "2024-01-01 00:01:00", 0.1, 0.2, 1000, 80, 0.5, 100.0, 200.0, 5, 10, 1.0, 0),
            (2, "2024-01-01 00:06:00", 0.1, 0.2, 1000, 80, 0.5, 100.0, 200.0, 5, 10, 1.0, 0),
        ],
    )
    conn.commit()
    try:
        agg_rows = layer2.aggregate_by_window(conn)
    finally:
        conn.close()
    assert agg_rows == 2


# ---------------------------------------------------------------------------
# flag_heavy_hitters
# ---------------------------------------------------------------------------
def test_flag_heavy_hitters_threshold_is_mean_plus_2sigma(load_layer, tmp_path):
    """
    Seed ip_window_aggregates with 5 hand-picked values and assert that
    flag_heavy_hitters flags exactly the rows strictly above mean+2*stdev.
    """
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "hh.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        DROP TABLE IF EXISTS ip_window_aggregates;
        CREATE TABLE ip_window_aggregates (
            source_ip_bin           REAL,
            window_start            TEXT,
            total_bytes_sent        REAL,
            total_bytes_received    REAL,
            connection_count        INTEGER,
            avg_duration            REAL,
            avg_packets_sent        REAL,
            max_bytes_sent          REAL,
            anomaly_count_in_window INTEGER
        );
    """)
    # Use 10 values with one outlier. The math has to be checked: with
    # 1 dominant value out of 10, threshold = mean + 2*stdev tends to
    # *equal* the outlier (mean ≈ outlier/10, stdev ≈ outlier). So the
    # outlier is exactly AT the threshold, not above. To make the test
    # work, we need a value strictly above that threshold -- which
    # means the outlier value itself must be > 2*10*outlier_approx,
    # which is impossible. So we use a value clearly far from the bulk
    # but not so extreme that it dominates the mean+2*stdev sum.
    #
    # A robust seed: bulk values around 100, one outlier at 1000.
    #   mean ≈ 200, stdev ≈ 270, threshold ≈ 740. 1000 is above 740.
    values = [100.0, 110.0, 90.0, 105.0, 95.0, 100.0, 110.0, 90.0, 105.0, 1000.0]
    expected_threshold = (sum(values) / len(values)
                          + layer2.HEAVY_HITTER_SIGMA
                          * math.sqrt(sum((v - sum(values) / len(values)) ** 2
                                          for v in values) / len(values)))
    expected_flagged = [v for v in values if v > expected_threshold]

    for i, v in enumerate(values):
        conn.execute(
            "INSERT INTO ip_window_aggregates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (float(i + 1), "2024-01-01 00:00:00", v, 0.0, 1, 0.0, 0.0, 0.0, 0),
        )
    conn.commit()

    flagged_count = layer2.flag_heavy_hitters(conn)
    conn.close()

    # Sanity-check our hand-computation picked up at least one row.
    assert len(expected_flagged) >= 1
    # The function must agree.
    assert flagged_count == len(expected_flagged)


def test_flag_heavy_hitters_no_flag_when_all_equal(load_layer, tmp_path):
    """If all rows have the same total_bytes_sent, stdev is 0 and
    threshold equals mean -- so nothing is strictly above the threshold."""
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "hh_flat.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        DROP TABLE IF EXISTS ip_window_aggregates;
        CREATE TABLE ip_window_aggregates (
            source_ip_bin REAL, window_start TEXT,
            total_bytes_sent REAL, total_bytes_received REAL,
            connection_count INTEGER, avg_duration REAL,
            avg_packets_sent REAL, max_bytes_sent REAL,
            anomaly_count_in_window INTEGER
        );
    """)
    for i in range(5):
        conn.execute(
            "INSERT INTO ip_window_aggregates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (float(i + 1), "2024-01-01 00:00:00", 100.0, 0.0, 1, 0.0, 0.0, 0.0, 0),
        )
    conn.commit()
    flagged = layer2.flag_heavy_hitters(conn)
    conn.close()
    assert flagged == 0


# ---------------------------------------------------------------------------
# create_processed_table
# ---------------------------------------------------------------------------
def test_create_processed_table_joins_features(load_layer, tmp_path):
    """End-to-end through layer 2: raw -> processed with all derived cols."""
    layer2 = load_layer("2_sql_transformations")
    db_path = tmp_path / "proc.db"
    conn = _seed_raw_traffic(db_path, n_normal=10, n_anomaly=1)
    try:
        layer2.filter_invalid_data(conn)
        layer2.aggregate_by_window(conn)
        layer2.flag_heavy_hitters(conn)
        processed = layer2.create_processed_table(conn)
    finally:
        conn.close()

    assert processed == 11  # 10 normal + 1 anomaly

    conn = sqlite3.connect(str(db_path))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(processed_traffic)").fetchall()]
    finally:
        conn.close()

    expected_new = {
        "PacketSize", "source_ip_bin", "window_total_bytes",
        "window_conn_count", "window_avg_duration", "window_anomaly_count",
        "is_heavy_hitter",
    }
    assert expected_new.issubset(set(cols)), \
        f"missing derived columns: {expected_new - set(cols)}"
