"""
=============================================================================
 Layer 4 (optional): Push SQLite -> Postgres
=============================================================================
 Reads the network_traffic.db SQLite file produced by the 3-layer pipeline
 and copies each table into a local Postgres database using the COPY command
 (fastest path for ~1M rows).

 Tables pushed (in this order, so FK-friendly if you add them later):
   - raw_traffic              (~1M rows)
   - ip_window_aggregates     (~162K rows)
   - heavy_hitters            (~5K rows)
   - processed_traffic        (~1M rows)
   - alerts                   (~50K rows)

 Usage:
   # Drop & recreate all tables (default)
   python push_to_postgres.py

   # Append to existing tables (must already exist with matching schema)
   python push_to_postgres.py --mode append

   # Push a subset of tables
   python push_to_postgres.py --tables alerts,heavy_hitters

   # Keep the intermediate CSVs on disk for debugging
   python push_to_postgres.py --keep-csvs

 Connection details come from environment variables:
   PGHOST     (default: localhost)
   PGPORT     (default: 5432)
   PGDATABASE (default: postgres)
   PGUSER     (default: postgres)
   PGPASSWORD (no default -- must be set)

 Key Decisions:
   - COPY via temp CSV is ~10-50x faster than row-by-row INSERT for 1M rows.
   - We use SQLite's `.iterdump()`-free approach: stream each table to CSV via
     pandas, then let Postgres COPY parse it. This keeps memory bounded.
   - Column types are inferred from the SQLite pragma and translated to
     Postgres equivalents. Timestamps (stored as TEXT in SQLite) become TIMESTAMP.
   - CSVs are written to a temp dir and deleted by default.
=============================================================================
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from typing import Optional

import pandas as pd
import psycopg2
import sqlite3

# Optional: auto-load .env from the project root if python-dotenv is installed.
# This is best-effort -- if dotenv is missing, env vars must be set in the shell.
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "network_traffic.db")

# Tables to push, in dependency-friendly order
ALL_TABLES = [
    "raw_traffic",
    "ip_window_aggregates",
    "heavy_hitters",
    "processed_traffic",
    "alerts",
]

# Map SQLite declared types -> Postgres types for columns the pipeline writes.
# SQLite is dynamically typed, so this maps the CREATE TABLE declarations.
SQLITE_TO_PG_TYPES = {
    "INTEGER": "BIGINT",
    "REAL":    "DOUBLE PRECISION",
    "TEXT":    "TEXT",
    "BLOB":    "BYTEA",
}


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
def print_header(mode: str) -> None:
    print("\n" + "=" * 70)
    print("  LAYER 4 -- SQLITE -> POSTGRES PUSH")
    print(f"  Mode: {mode.upper()}")
    print("=" * 70)


def step(msg: str) -> None:
    print(f"\n  >> {msg}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def get_pg_config() -> dict:
    """Read Postgres connection config from environment variables."""
    cfg = {
        "host":     os.environ.get("PGHOST", "localhost"),
        "port":     int(os.environ.get("PGPORT", "5432")),
        "dbname":   os.environ.get("PGDATABASE", "postgres"),
        "user":     os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD"),
    }
    if cfg["password"] is None:
        print("  [X] ERROR: PGPASSWORD environment variable is not set.")
        print("    Either set it in your shell, or add it to a .env file in this")
        print("    directory (requires `pip install python-dotenv` for auto-loading):")
        print("      export PGPASSWORD=your_password        (bash)")
        print("      $env:PGPASSWORD = 'your_password'       (PowerShell)")
        print("      PGPASSWORD=your_password               (in .env)")
        sys.exit(1)
    return cfg


def connect_pg(cfg: dict) -> psycopg2.extensions.connection:
    """Open a Postgres connection and verify it works."""
    try:
        conn = psycopg2.connect(**cfg)
    except psycopg2.OperationalError as e:
        print(f"  [X] ERROR: Could not connect to Postgres at {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
        print(f"    {e}")
        sys.exit(1)
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
    print(f"  [OK] Connected to Postgres: {version}")
    return conn


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    if not os.path.isfile(db_path):
        print(f"  [X] ERROR: SQLite database not found: {db_path}")
        print("    Run run_pipeline.py first to generate it.")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    n_tables = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    print(f"  [OK] Connected to SQLite ({n_tables} tables in {os.path.basename(db_path)})")
    return conn


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
def get_table_schema(sqlite_conn: sqlite3.Connection, table: str) -> list:
    """Return [(column_name, sqlite_declared_type), ...] for a table."""
    rows = sqlite_conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    return [(r[1], (r[2] or "TEXT").upper()) for r in rows]


def get_row_count(sqlite_conn: sqlite3.Connection, table: str) -> int:
    return sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def build_create_statement(table: str, schema: list, if_not_exists: bool = False) -> str:
    """
    Build a Postgres CREATE TABLE statement from a SQLite schema.
    Timestamps (column name == 'timestamp') are stored as TIMESTAMP rather than TEXT
    so COPY can parse them as proper timestamps.
    """
    parts = []
    for col_name, sqlite_type in schema:
        if col_name == "timestamp":
            pg_type = "TIMESTAMP"
        else:
            pg_type = SQLITE_TO_PG_TYPES.get(sqlite_type, "TEXT")
        parts.append(f'    "{col_name}" {pg_type}')

    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    return f"CREATE TABLE {exists_clause}{table} (\n" + ",\n".join(parts) + "\n);"


# ---------------------------------------------------------------------------
# Push logic
# ---------------------------------------------------------------------------
def stream_table_to_csv(sqlite_conn: sqlite3.Connection, table: str,
                        csv_path: str, schema: list) -> int:
    """
    Stream a SQLite table to a CSV file using pandas (in chunks to bound memory).
    Returns the number of rows written.
    """
    cols = [c for c, _ in schema]
    col_list = ", ".join(f'"{c}"' for c in cols)
    chunk_iter = pd.read_sql_query(
        f"SELECT {col_list} FROM {table}", sqlite_conn, chunksize=50_000
    )
    total = 0
    first = True
    for chunk in chunk_iter:
        # Normalize NaN -> empty string so COPY's CSV parser is happy.
        # (Postgres COPY treats empty as NULL by default with NULL ''.)
        chunk = chunk.where(pd.notnull(chunk), None)
        chunk.to_csv(
            csv_path,
            mode="w" if first else "a",
            index=False,
            header=first,
            quoting=csv.QUOTE_MINIMAL,
            na_rep="",
        )
        first = False
        total += len(chunk)
    return total


def copy_csv_to_pg(pg_conn, table: str, csv_path: str, columns: list) -> None:
    """Run COPY ... FROM ... CSV HEADER on a single file."""
    col_list = ", ".join(f'"{c}"' for c in columns)
    sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    with pg_conn.cursor() as cur, open(csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)
    pg_conn.commit()


def push_table(sqlite_conn: sqlite3.Connection, pg_conn, table: str,
               mode: str, tmp_dir: str) -> int:
    """Push a single table end-to-end. Returns the number of rows pushed."""
    step(f"Pushing table: {table}")
    schema = get_table_schema(sqlite_conn, table)
    n_rows = get_row_count(sqlite_conn, table)

    if n_rows == 0:
        print(f"  |-- Table {table} is empty -- skipping")
        return 0

    with pg_conn.cursor() as cur:
        if mode == "drop":
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            cur.execute(build_create_statement(table, schema))
            pg_conn.commit()
            print(f"  |-- Recreated schema ({len(schema)} columns)")
        else:  # append
            exists = cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s",
                (table,),
            ).fetchone()
            if not exists:
                print(f"  [X] ERROR: --mode append but table '{table}' does not exist in Postgres.")
                print(f"    Run with --mode drop first, or create the table manually.")
                sys.exit(1)
            print(f"  |-- Appending to existing table")

    # Stream SQLite -> CSV -> COPY
    csv_path = os.path.join(tmp_dir, f"{table}.csv")
    t0 = time.time()
    written = stream_table_to_csv(sqlite_conn, table, csv_path, schema)
    csv_time = time.time() - t0

    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    t0 = time.time()
    copy_csv_to_pg(pg_conn, table, csv_path, [c for c, _ in schema])
    copy_time = time.time() - t0

    # Verify row count
    with pg_conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        pg_count = cur.fetchone()[0]

    print(f"  |-- SQLite rows:  {written:>12,}")
    print(f"  |-- Postgres rows:{pg_count:>12,}  (CSV: {size_mb:.1f} MB, "
          f"export {csv_time:.1f}s + load {copy_time:.1f}s)")
    if pg_count != written:
        print(f"  [WARN] Row count mismatch on {table}! "
              f"Expected {written:,}, found {pg_count:,}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push network_traffic.db tables into a local Postgres database."
    )
    p.add_argument(
        "--mode", choices=["drop", "append"], default="drop",
        help="drop = drop and recreate each table; append = require pre-existing tables.",
    )
    p.add_argument(
        "--tables", default=",".join(ALL_TABLES),
        help=f"Comma-separated list of tables to push. Default: all ({','.join(ALL_TABLES)}).",
    )
    p.add_argument(
        "--keep-csvs", action="store_true",
        help="Keep the intermediate CSV files on disk for debugging.",
    )
    p.add_argument(
        "--csv-dir", default=None,
        help="Directory for intermediate CSVs (default: a fresh temp dir).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in tables if t not in ALL_TABLES]
    if unknown:
        print(f"  [X] ERROR: Unknown table(s): {unknown}")
        print(f"    Valid options: {ALL_TABLES}")
        return 1

    print_header(args.mode)

    pg_cfg = get_pg_config()
    sqlite_conn = connect_sqlite(DB_PATH)
    pg_conn = connect_pg(pg_cfg)

    tmp_dir = args.csv_dir or tempfile.mkdtemp(prefix="sqlite_to_pg_")
    if args.csv_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    print(f"  [OK] Using temp directory: {tmp_dir}")

    total_rows = 0
    pipeline_start = time.time()
    try:
        for table in tables:
            total_rows += push_table(sqlite_conn, pg_conn, table, args.mode, tmp_dir)
    finally:
        sqlite_conn.close()
        pg_conn.close()
        if not args.keep_csvs and not args.csv_dir:
            # Clean up temp CSVs we created
            for t in tables:
                p = os.path.join(tmp_dir, f"{t}.csv")
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass  # not empty, leave it
        elif args.keep_csvs:
            print(f"  [OK] Kept CSVs in: {tmp_dir}")

    elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print(f"  [DONE] PUSH COMPLETE -- {total_rows:,} total rows in {elapsed:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
