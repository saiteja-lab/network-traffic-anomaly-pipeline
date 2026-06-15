"""
=============================================================================
 Layer 2: SQL Transformation -- Feature Engineering
=============================================================================
 Runs SQL queries against the raw_traffic table to:
   1. FILTER  -- Remove nulls and extreme protocol outliers
   2. AGGREGATE -- Calculate total bytes per IP-bin over 5-minute windows
   3. FLAG -- Identify "Heavy Hitter" IPs that exceed a threshold

 Creates a cleaned + enriched `processed_traffic` table for the AI model.

 Key Decisions:
   - SQLite lacks STDEV(), so we compute it in Python and inject as a param.
   - SourceIP is binned (rounded to 1 decimal) since values are z-score floats.
   - 5-minute windows are simulated using timestamp arithmetic.
=============================================================================
"""

import os
import sys
import time
import math
import sqlite3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "network_traffic.db")

# Rows with |Protocol| > this are considered invalid / extreme outliers
PROTOCOL_OUTLIER_THRESHOLD = 3.0

# Heavy hitter = total_bytes > mean + HEAVY_HITTER_SIGMA * stdev
HEAVY_HITTER_SIGMA = 2.0


def print_header():
    """Print a styled header for the transformation step."""
    print("\n" + "=" * 70)
    print("  LAYER 2 -- SQL TRANSFORMATIONS (Feature Engineering)")
    print("  Raw Traffic  ->  Processed Features")
    print("=" * 70)


def connect_db(db_path: str) -> sqlite3.Connection:
    """Open the database and verify raw_traffic exists."""
    if not os.path.isfile(db_path):
        print(f"  [X] ERROR: Database not found: {db_path}")
        print("    Run 1_data_ingestion.py first.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # Verify the table exists
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='raw_traffic'"
    )
    if cursor.fetchone()[0] == 0:
        print("  [X] ERROR: raw_traffic table not found in database.")
        sys.exit(1)

    cursor = conn.execute("SELECT COUNT(*) FROM raw_traffic")
    count = cursor.fetchone()[0]
    print(f"  [OK] Connected to database ({count:,} rows in raw_traffic)")
    return conn


# ---------------------------------------------------------------------------
# Transformation 1: FILTERING -- Clean invalid data
# ---------------------------------------------------------------------------
def filter_invalid_data(conn: sqlite3.Connection) -> int:
    """
    Remove rows with:
      - Any NULL values in key columns
      - Protocol outliers (|Protocol| > threshold)

    Returns the number of rows removed.
    """
    print("\n  -- Step 1: Filtering Invalid Data -------------------------")

    before = conn.execute("SELECT COUNT(*) FROM raw_traffic").fetchone()[0]

    # Count NULLs
    null_count = conn.execute("""
        SELECT COUNT(*) FROM raw_traffic
        WHERE SourceIP IS NULL
           OR DestinationIP IS NULL
           OR BytesSent IS NULL
           OR BytesReceived IS NULL
           OR PacketsSent IS NULL
           OR PacketsReceived IS NULL
           OR Duration IS NULL
           OR Protocol IS NULL
    """).fetchone()[0]

    # Count protocol outliers
    outlier_count = conn.execute("""
        SELECT COUNT(*) FROM raw_traffic
        WHERE ABS(Protocol) > ?
    """, (PROTOCOL_OUTLIER_THRESHOLD,)).fetchone()[0]

    # Remove NULLs
    conn.execute("""
        DELETE FROM raw_traffic
        WHERE SourceIP IS NULL
           OR DestinationIP IS NULL
           OR BytesSent IS NULL
           OR BytesReceived IS NULL
           OR PacketsSent IS NULL
           OR PacketsReceived IS NULL
           OR Duration IS NULL
           OR Protocol IS NULL
    """)

    # Remove protocol outliers
    conn.execute("""
        DELETE FROM raw_traffic
        WHERE ABS(Protocol) > ?
    """, (PROTOCOL_OUTLIER_THRESHOLD,))

    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM raw_traffic").fetchone()[0]
    removed = before - after

    print(f"  | Rows with NULL values:         {null_count:>10,}")
    print(f"  | Rows with |Protocol| > {PROTOCOL_OUTLIER_THRESHOLD}:    {outlier_count:>10,}")
    print(f"  | Total rows removed:            {removed:>10,}")
    print(f"  | Remaining rows:                {after:>10,}")
    print(f"  -----------------------------------------------------------")
    return removed


# ---------------------------------------------------------------------------
# Transformation 2: AGGREGATION -- 5-minute sliding window
# ---------------------------------------------------------------------------
def aggregate_by_window(conn: sqlite3.Connection) -> int:
    """
    Calculate total bytes sent per SourceIP bin over 5-minute windows.
    Creates the `ip_window_aggregates` table.

    Returns the number of aggregate rows created.
    """
    print("\n  -- Step 2: Aggregation (5-Minute Windows) -----------------")

    conn.execute("DROP TABLE IF EXISTS ip_window_aggregates")
    conn.execute("""
        CREATE TABLE ip_window_aggregates AS
        SELECT
            ROUND(SourceIP, 1)  AS source_ip_bin,

            /* Floor timestamp to nearest 5-minute boundary */
            strftime('%Y-%m-%d %H:%M:00',
                timestamp,
                '-' || (CAST(strftime('%M', timestamp) AS INTEGER) % 5) || ' minutes'
            ) AS window_start,

            SUM(BytesSent)      AS total_bytes_sent,
            SUM(BytesReceived)  AS total_bytes_received,
            COUNT(*)            AS connection_count,
            AVG(Duration)       AS avg_duration,
            AVG(PacketsSent)    AS avg_packets_sent,
            MAX(BytesSent)      AS max_bytes_sent,
            SUM(IsAnomaly)      AS anomaly_count_in_window
        FROM raw_traffic
        GROUP BY source_ip_bin, window_start
    """)
    conn.commit()

    agg_count = conn.execute("SELECT COUNT(*) FROM ip_window_aggregates").fetchone()[0]
    unique_ips = conn.execute(
        "SELECT COUNT(DISTINCT source_ip_bin) FROM ip_window_aggregates"
    ).fetchone()[0]
    unique_windows = conn.execute(
        "SELECT COUNT(DISTINCT window_start) FROM ip_window_aggregates"
    ).fetchone()[0]

    print(f"  | Aggregate rows created:        {agg_count:>10,}")
    print(f"  | Unique IP bins:                {unique_ips:>10,}")
    print(f"  | Unique time windows:           {unique_windows:>10,}")

    # Show a sample
    sample = conn.execute("""
        SELECT source_ip_bin, window_start, total_bytes_sent, connection_count
        FROM ip_window_aggregates
        ORDER BY total_bytes_sent DESC
        LIMIT 5
    """).fetchall()
    print(f"  |")
    print(f"  | Top 5 by total bytes sent:")
    print(f"  |   {'IP Bin':>8}  {'Window Start':>20}  {'Bytes':>12}  {'Conns':>6}")
    for row in sample:
        print(f"  |   {row[0]:>8.1f}  {row[1]:>20}  {row[2]:>12.2f}  {row[3]:>6}")
    print(f"  -----------------------------------------------------------")

    return agg_count


# ---------------------------------------------------------------------------
# Transformation 3: FLAGGING -- Heavy Hitters
# ---------------------------------------------------------------------------
def flag_heavy_hitters(conn: sqlite3.Connection) -> int:
    """
    Identify IP bins whose total_bytes_sent exceeds
    mean + HEAVY_HITTER_SIGMA * stdev.

    Creates the `heavy_hitters` table.
    Returns the number of heavy hitters found.
    """
    print("\n  -- Step 3: Flagging Heavy Hitters --------------------------")

    # Compute mean and stdev of total_bytes_sent in Python
    # (SQLite doesn't have a built-in STDEV function)
    cursor = conn.execute("""
        SELECT total_bytes_sent FROM ip_window_aggregates
    """)
    values = [row[0] for row in cursor.fetchall()]
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n
    stdev_val = math.sqrt(variance)
    threshold = mean_val + HEAVY_HITTER_SIGMA * stdev_val

    print(f"  | Mean total_bytes_sent:         {mean_val:>12.4f}")
    print(f"  | Stdev:                         {stdev_val:>12.4f}")
    print(f"  | Threshold (mean + {HEAVY_HITTER_SIGMA}*std):    {threshold:>12.4f}")

    # Create heavy hitters table
    conn.execute("DROP TABLE IF EXISTS heavy_hitters")
    conn.execute("""
        CREATE TABLE heavy_hitters AS
        SELECT
            source_ip_bin,
            window_start,
            total_bytes_sent,
            total_bytes_received,
            connection_count,
            avg_duration,
            anomaly_count_in_window,
            1 AS is_heavy_hitter
        FROM ip_window_aggregates
        WHERE total_bytes_sent > ?
    """, (threshold,))
    conn.commit()

    hh_count = conn.execute("SELECT COUNT(*) FROM heavy_hitters").fetchone()[0]
    hh_pct = hh_count / n * 100 if n > 0 else 0

    print(f"  | Heavy hitters flagged:         {hh_count:>10,} ({hh_pct:.2f}%)")

    # Show top heavy hitters
    sample = conn.execute("""
        SELECT source_ip_bin, window_start, total_bytes_sent, connection_count
        FROM heavy_hitters
        ORDER BY total_bytes_sent DESC
        LIMIT 5
    """).fetchall()

    if sample:
        print(f"  |")
        print(f"  | Top 5 heavy hitters:")
        print(f"  |   {'IP Bin':>8}  {'Window Start':>20}  {'Bytes':>12}  {'Conns':>6}")
        for row in sample:
            print(f"  |   {row[0]:>8.1f}  {row[1]:>20}  {row[2]:>12.2f}  {row[3]:>6}")

    print(f"  -----------------------------------------------------------")
    return hh_count


# ---------------------------------------------------------------------------
# Final: Create processed_traffic table
# ---------------------------------------------------------------------------
def create_processed_table(conn: sqlite3.Connection) -> int:
    """
    Join the cleaned raw_traffic data with window aggregates to create
    the final processed_traffic table for the AI model.

    Returns the number of rows in the processed table.
    """
    print("\n  -- Step 4: Creating Processed Table ------------------------")

    conn.execute("DROP TABLE IF EXISTS processed_traffic")
    conn.execute("""
        CREATE TABLE processed_traffic AS
        SELECT
            r.row_id,
            r.timestamp,
            r.SourceIP,
            r.DestinationIP,
            r.SourcePort,
            r.DestinationPort,
            r.Protocol,
            r.BytesSent,
            r.BytesReceived,
            r.PacketsSent,
            r.PacketsReceived,
            r.Duration,
            r.IsAnomaly,

            /* Derived features */
            (r.BytesSent + r.BytesReceived) AS PacketSize,
            ROUND(r.SourceIP, 1) AS source_ip_bin,

            /* Aggregated window features (joined) */
            COALESCE(a.total_bytes_sent, 0)       AS window_total_bytes,
            COALESCE(a.connection_count, 0)        AS window_conn_count,
            COALESCE(a.avg_duration, 0)            AS window_avg_duration,
            COALESCE(a.anomaly_count_in_window, 0) AS window_anomaly_count,

            /* Heavy hitter flag */
            CASE WHEN h.is_heavy_hitter = 1 THEN 1 ELSE 0 END AS is_heavy_hitter

        FROM raw_traffic r
        LEFT JOIN ip_window_aggregates a
            ON ROUND(r.SourceIP, 1) = a.source_ip_bin
            AND strftime('%Y-%m-%d %H:%M:00',
                    r.timestamp,
                    '-' || (CAST(strftime('%M', r.timestamp) AS INTEGER) % 5) || ' minutes'
                ) = a.window_start
        LEFT JOIN heavy_hitters h
            ON ROUND(r.SourceIP, 1) = h.source_ip_bin
            AND strftime('%Y-%m-%d %H:%M:00',
                    r.timestamp,
                    '-' || (CAST(strftime('%M', r.timestamp) AS INTEGER) % 5) || ' minutes'
                ) = h.window_start
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM processed_traffic").fetchone()[0]

    # Show schema
    cols = conn.execute("PRAGMA table_info(processed_traffic)").fetchall()

    print(f"  | Rows in processed_traffic:     {count:>10,}")
    print(f"  | Columns:                       {len(cols):>10}")
    print(f"  |")
    print(f"  | New derived columns:")
    print(f"  |   - PacketSize            (BytesSent + BytesReceived)")
    print(f"  |   - source_ip_bin         (SourceIP rounded to 0.1)")
    print(f"  |   - window_total_bytes    (from 5-min aggregation)")
    print(f"  |   - window_conn_count     (connections in window)")
    print(f"  |   - window_avg_duration   (avg duration in window)")
    print(f"  |   - window_anomaly_count  (anomalies in window)")
    print(f"  |   - is_heavy_hitter       (1 if above threshold)")
    print(f"  -----------------------------------------------------------")

    return count


def run():
    """Execute all SQL transformations."""
    print_header()
    start = time.time()

    print("\n  [Connecting to database...]")
    conn = connect_db(DB_PATH)

    rows_removed = filter_invalid_data(conn)
    agg_rows = aggregate_by_window(conn)
    hh_count = flag_heavy_hitters(conn)
    processed_count = create_processed_table(conn)

    elapsed = time.time() - start

    print(f"\n  [DONE] TRANSFORMATIONS COMPLETE in {elapsed:.1f}s")
    print(f"  | Rows filtered out:    {rows_removed:>10,}")
    print(f"  | Aggregate rows:       {agg_rows:>10,}")
    print(f"  | Heavy hitters:        {hh_count:>10,}")
    print(f"  | Processed rows:       {processed_count:>10,}")

    conn.close()
    return processed_count


if __name__ == "__main__":
    run()
