"""
=============================================================================
 Layer 1: Data Ingestion -- CSV to SQLite ETL Pipeline
=============================================================================
 Reads the synthetic_network_traffic.csv file and loads it into a SQLite
 database (network_traffic.db). Uses chunked loading to handle the ~190MB
 file efficiently without exhausting memory.

 Key Decisions:
   - Chunked pandas read (50K rows/chunk) keeps peak memory ~60MB
   - Synthetic timestamps are generated to enable temporal SQL queries
   - A row_id primary key is added for referential integrity
=============================================================================
"""

import os
import sys
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "synthetic_network_traffic.csv")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "network_traffic.db")
CHUNK_SIZE = 50_000
START_TIMESTAMP = datetime(2024, 1, 1, 0, 0, 0)


def print_header():
    """Print a styled header for the ETL step."""
    print("\n" + "=" * 70)
    print("  LAYER 1 -- DATA INGESTION (ETL)")
    print("  CSV  ->  SQLite Database")
    print("=" * 70)


def validate_csv(csv_path: str) -> None:
    """Check that the source CSV exists and is readable."""
    if not os.path.isfile(csv_path):
        print(f"  [X] ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  [OK] Source CSV found: {os.path.basename(csv_path)} ({size_mb:.1f} MB)")


def create_database(db_path: str) -> sqlite3.Connection:
    """Create (or overwrite) the SQLite database and return a connection."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  [OK] Removed existing database: {os.path.basename(db_path)}")

    conn = sqlite3.connect(db_path)
    print(f"  [OK] Created database: {os.path.basename(db_path)}")
    return conn


def create_table(conn: sqlite3.Connection) -> None:
    """Create the raw_traffic table with the expected schema."""
    conn.execute("DROP TABLE IF EXISTS raw_traffic")
    conn.execute("""
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
        )
    """)
    conn.commit()
    print("  [OK] Created table: raw_traffic")


def ingest_csv(csv_path: str, conn: sqlite3.Connection) -> int:
    """
    Read the CSV in chunks, add synthetic timestamps + row IDs,
    and insert into the raw_traffic table.

    Returns the total number of rows ingested.
    """
    total_rows = 0
    row_id_offset = 0

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

    for chunk_num, chunk in enumerate(reader, start=1):
        num_rows = len(chunk)

        # Generate sequential row IDs
        chunk.insert(0, "row_id", range(row_id_offset + 1, row_id_offset + num_rows + 1))

        # Generate synthetic timestamps (1 second apart)
        timestamps = [
            (START_TIMESTAMP + timedelta(seconds=row_id_offset + i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(num_rows)
        ]
        chunk.insert(1, "timestamp", timestamps)

        # Write to SQLite
        chunk.to_sql("raw_traffic", conn, if_exists="append", index=False)

        row_id_offset += num_rows
        total_rows += num_rows

        # Progress indicator
        print(f"\r  [..] Loading chunk {chunk_num:>3} | "
              f"{total_rows:>10,} rows ingested", end="", flush=True)

    print()  # newline after progress
    return total_rows


def verify_ingestion(conn: sqlite3.Connection) -> None:
    """Run verification queries and print summary statistics."""
    cursor = conn.cursor()

    # Row count
    cursor.execute("SELECT COUNT(*) FROM raw_traffic")
    count = cursor.fetchone()[0]

    # Time range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM raw_traffic")
    min_ts, max_ts = cursor.fetchone()

    # Anomaly distribution
    cursor.execute("""
        SELECT IsAnomaly, COUNT(*) as cnt
        FROM raw_traffic
        GROUP BY IsAnomaly
        ORDER BY IsAnomaly
    """)
    anomaly_dist = cursor.fetchall()

    # Schema info
    cursor.execute("PRAGMA table_info(raw_traffic)")
    columns = cursor.fetchall()

    print("\n  -- Verification -----------------------------------------------")
    print(f"  | Total rows loaded:  {count:>12,}")
    print(f"  | Timestamp range:    {min_ts}  ->  {max_ts}")
    print(f"  | Table columns:      {len(columns)}")
    print(f"  |")
    for label, cnt in anomaly_dist:
        pct = cnt / count * 100
        print(f"  | IsAnomaly={label}:       {cnt:>10,} ({pct:.2f}%)")
    print(f"  ---------------------------------------------------------------")


def run():
    """Execute the full ingestion pipeline."""
    print_header()
    start = time.time()

    print("\n  [Step 1/4] Validating source file...")
    validate_csv(CSV_PATH)

    print("\n  [Step 2/4] Creating SQLite database...")
    conn = create_database(DB_PATH)

    print("\n  [Step 3/4] Creating table schema...")
    create_table(conn)

    print("\n  [Step 4/4] Ingesting CSV data (chunked)...")
    total = ingest_csv(CSV_PATH, conn)

    verify_ingestion(conn)

    # Create index for faster queries in Layer 2
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON raw_traffic(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_ip ON raw_traffic(SourceIP)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly ON raw_traffic(IsAnomaly)")
    conn.commit()
    print("  [OK] Created indexes on timestamp, SourceIP, IsAnomaly")

    elapsed = time.time() - start
    print(f"\n  [DONE] INGESTION COMPLETE -- {total:,} rows in {elapsed:.1f}s")
    print(f"  Database: {DB_PATH}")

    conn.close()
    return total


if __name__ == "__main__":
    run()
