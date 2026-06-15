"""
=============================================================================
 Network Traffic Anomaly Detection -- Full Pipeline Runner
=============================================================================
 Master script that executes all 3 layers in sequence:
   1. Data Ingestion  -- CSV -> SQLite
   2. SQL Transforms  -- Feature engineering
   3. AI Detection    -- Isolation Forest anomaly detection

 Usage:
   python run_pipeline.py
=============================================================================
"""

import time
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the main pipeline banner."""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|   NETWORK TRAFFIC ANOMALY DETECTION PIPELINE" + " " * 23 + "|")
    print("|" + " " * 68 + "|")
    print("|   Layers:" + " " * 58 + "|")
    print("|     1. Data Ingestion  (CSV -> SQLite)" + " " * 29 + "|")
    print("|     2. SQL Transforms  (Feature Engineering)" + " " * 22 + "|")
    print("|     3. AI Detection    (Isolation Forest)" + " " * 25 + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")


def print_summary(total_time: float, results: dict):
    """Print the final pipeline summary."""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|   PIPELINE EXECUTION SUMMARY" + " " * 39 + "|")
    print("|" + " " * 68 + "|")

    lines = [
        f"Total execution time:     {total_time:.1f}s",
        f"",
        f"Layer 1 -- Ingestion:",
        f"  Rows ingested:          {results.get('ingested', 'N/A'):>12,}" if isinstance(results.get('ingested'), int) else f"  Rows ingested:          {'N/A':>12}",
        f"",
        f"Layer 2 -- Transformations:",
        f"  Processed rows:         {results.get('processed', 'N/A'):>12,}" if isinstance(results.get('processed'), int) else f"  Processed rows:         {'N/A':>12}",
        f"",
        f"Layer 3 -- AI Detection:",
        f"  Anomalies detected:     {results.get('alerts', 'N/A'):>12,}" if isinstance(results.get('alerts'), int) else f"  Anomalies detected:     {'N/A':>12}",
    ]

    for line in lines:
        padded = f"|   {line}"
        padded += " " * (69 - len(padded)) + "|"
        print(padded)

    print("|" + " " * 68 + "|")
    print("|   [DONE] ALL LAYERS COMPLETED SUCCESSFULLY" + " " * 25 + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print()


def main():
    """Run the full 3-layer pipeline."""
    print_banner()
    pipeline_start = time.time()
    results = {}

    # -- Layer 1: Data Ingestion --
    try:
        import importlib
        layer1 = importlib.import_module("1_data_ingestion")
        results["ingested"] = layer1.run()
    except Exception as e:
        print(f"\n  [X] LAYER 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Layer 2: SQL Transformations --
    try:
        layer2 = importlib.import_module("2_sql_transformations")
        results["processed"] = layer2.run()
    except Exception as e:
        print(f"\n  [X] LAYER 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Layer 3: AI Detection --
    try:
        layer3 = importlib.import_module("3_anomaly_detection")
        results["alerts"] = layer3.run()
    except Exception as e:
        print(f"\n  [X] LAYER 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Summary --
    total_time = time.time() - pipeline_start
    print_summary(total_time, results)


if __name__ == "__main__":
    main()
