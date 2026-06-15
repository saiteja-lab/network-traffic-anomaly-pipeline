"""
=============================================================================
 Layer 3: AI Detection Engine -- Isolation Forest Anomaly Detection
=============================================================================
 Loads the processed_traffic table from SQLite, trains an Isolation Forest
 model on PacketSize and Duration, and generates anomaly alerts.

 Key Decisions:
   - Isolation Forest is ideal for unsupervised anomaly detection
   - We train on PacketSize (BytesSent + BytesReceived) and Duration
   - We validate against the ground truth IsAnomaly column
   - Results saved back to SQLite `alerts` table + scatter plot PNG
=============================================================================
"""

import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "network_traffic.db")
CHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "anomaly_results.png")

# Isolation Forest hyperparameters
CONTAMINATION = 0.05       # Expected fraction of anomalies
N_ESTIMATORS = 100         # Number of trees
RANDOM_STATE = 42          # Reproducibility
MAX_SAMPLES = 50_000       # Subsample for training efficiency on 1M rows


def print_header():
    """Print a styled header for the AI detection step."""
    print("\n" + "=" * 70)
    print("  LAYER 3 -- AI ANOMALY DETECTION ENGINE")
    print("  Isolation Forest  ->  Alerts")
    print("=" * 70)


def load_data(db_path: str) -> pd.DataFrame:
    """Load the processed_traffic table from SQLite into a DataFrame."""
    if not os.path.isfile(db_path):
        print(f"  [X] ERROR: Database not found: {db_path}")
        print("    Run 1_data_ingestion.py and 2_sql_transformations.py first.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # Check table exists
    exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='processed_traffic'"
    ).fetchone()[0]
    if not exists:
        print("  [X] ERROR: processed_traffic table not found.")
        print("    Run 2_sql_transformations.py first.")
        sys.exit(1)

    df = pd.read_sql_query("SELECT * FROM processed_traffic", conn)
    conn.close()

    print(f"  [OK] Loaded {len(df):,} rows from processed_traffic")
    return df


def train_isolation_forest(df: pd.DataFrame) -> tuple:
    """
    Train an Isolation Forest model on PacketSize and Duration.

    Returns:
        (model, predictions, anomaly_scores)
        predictions: -1 = anomaly, 1 = normal
    """
    print("\n  -- Training Isolation Forest -------------------------------")

    # Feature matrix
    features = df[["PacketSize", "Duration"]].copy()
    features = features.fillna(0)

    print(f"  | Features:      PacketSize, Duration")
    print(f"  | Training rows: {len(features):>10,}")
    print(f"  | Contamination: {CONTAMINATION}")
    print(f"  | Estimators:    {N_ESTIMATORS}")
    print(f"  | Max samples:   {MAX_SAMPLES:,}")

    # Train
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        max_samples=min(MAX_SAMPLES, len(features)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(features)
    train_time = time.time() - t0

    # Predict
    predictions = model.predict(features)          # -1 = anomaly, 1 = normal
    anomaly_scores = model.decision_function(features)  # lower = more anomalous

    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()

    print(f"  |")
    print(f"  | Training time:    {train_time:.2f}s")
    print(f"  | Normal detected:  {n_normal:>10,}")
    print(f"  | Anomalies found:  {n_anomalies:>10,} ({n_anomalies / len(features) * 100:.2f}%)")
    print(f"  -----------------------------------------------------------")

    return model, predictions, anomaly_scores


def evaluate_model(df: pd.DataFrame, predictions: np.ndarray) -> None:
    """
    Compare Isolation Forest predictions against ground truth IsAnomaly.
    Print precision, recall, F1, and confusion matrix.
    """
    print("\n  -- Model Evaluation (vs Ground Truth) ---------------------")

    # Convert predictions: IF returns -1=anomaly, 1=normal
    # Ground truth: 1=anomaly, 0=normal
    pred_labels = (predictions == -1).astype(int)  # 1 = anomaly, 0 = normal
    true_labels = df["IsAnomaly"].values

    # Metrics
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    print(f"  |")
    print(f"  |  Precision:  {precision:.4f}  (of predicted anomalies, how many were real)")
    print(f"  |  Recall:     {recall:.4f}  (of real anomalies, how many did we catch)")
    print(f"  |  F1 Score:   {f1:.4f}")
    print(f"  |")
    print(f"  |  Confusion Matrix:")
    print(f"  |                    Predicted Normal   Predicted Anomaly")
    print(f"  |  Actual Normal:    {tn:>15,}   {fp:>17,}")
    print(f"  |  Actual Anomaly:   {fn:>15,}   {tp:>17,}")
    print(f"  |")
    print(f"  |  True Positives:   {tp:>10,}  (correctly caught anomalies)")
    print(f"  |  False Positives:  {fp:>10,}  (false alarms)")
    print(f"  |  False Negatives:  {fn:>10,}  (missed anomalies)")
    print(f"  |  True Negatives:   {tn:>10,}  (correctly identified normal)")
    print(f"  -----------------------------------------------------------")

    # Full classification report
    print("\n  -- Detailed Classification Report -------------------------")
    report = classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
    for line in report.split("\n"):
        print(f"  | {line}")
    print(f"  -----------------------------------------------------------")


def generate_alerts(df: pd.DataFrame, predictions: np.ndarray,
                    anomaly_scores: np.ndarray, db_path: str) -> int:
    """
    Create an alerts table in SQLite with the most anomalous connections.
    Returns the number of alerts generated.
    """
    print("\n  -- Generating Alerts ------------------------------------------")

    # Filter to anomalies only
    anomaly_mask = predictions == -1
    alerts_df = df[anomaly_mask].copy()
    alerts_df["anomaly_score"] = anomaly_scores[anomaly_mask]
    alerts_df["prediction"] = -1

    # Sort by anomaly score (most anomalous first)
    alerts_df = alerts_df.sort_values("anomaly_score", ascending=True)

    # Select key columns for the alerts table
    alert_cols = [
        "row_id", "timestamp", "SourceIP", "DestinationIP",
        "PacketSize", "Duration", "BytesSent", "BytesReceived",
        "Protocol", "IsAnomaly", "is_heavy_hitter",
        "anomaly_score", "prediction"
    ]
    alerts_output = alerts_df[alert_cols].copy()

    # Save to SQLite
    conn = sqlite3.connect(db_path)
    alerts_output.to_sql("alerts", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"  | Total alerts generated:       {len(alerts_output):>10,}")
    print(f"  | Saved to table: alerts")

    # Print top 20 most anomalous
    print(f"  |")
    print(f"  | === TOP 20 MOST ANOMALOUS CONNECTIONS ===")
    print(f"  |")
    print(f"  | {'#':>3}  {'Timestamp':>20}  {'SrcIP':>8}  {'PktSize':>9}  "
          f"{'Duration':>9}  {'Score':>8}  {'Real':>4}")
    print(f"  | {'---':>3}  {'--------------------':>20}  {'--------':>8}  {'---------':>9}  "
          f"{'---------':>9}  {'--------':>8}  {'----':>4}")

    for i, (_, row) in enumerate(alerts_output.head(20).iterrows(), 1):
        real_flag = "!! " if row["IsAnomaly"] == 1 else "   "
        print(f"  | {i:>3}  {row['timestamp']:>20}  {row['SourceIP']:>8.3f}  "
              f"{row['PacketSize']:>9.3f}  {row['Duration']:>9.3f}  "
              f"{row['anomaly_score']:>8.4f}  {real_flag}")

    print(f"  |")
    print(f"  | (!! = actual anomaly in ground truth)")
    print(f"  -----------------------------------------------------------")

    return len(alerts_output)


def create_visualization(df: pd.DataFrame, predictions: np.ndarray,
                         anomaly_scores: np.ndarray, chart_path: str) -> None:
    """
    Generate a scatter plot of PacketSize vs Duration, colored by
    anomaly detection results. Saves to a PNG file.
    """
    print("\n  -- Creating Visualization -------------------------------------")

    # Subsample for plotting (1M points would be unreadable)
    sample_size = min(20_000, len(df))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(df), sample_size, replace=False)

    sample_df = df.iloc[sample_idx].copy()
    sample_preds = predictions[sample_idx]
    sample_scores = anomaly_scores[sample_idx]

    # Set up the figure
    sns.set_theme(style="darkgrid", palette="deep")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Network Traffic Anomaly Detection - Isolation Forest Results",
                 fontsize=16, fontweight="bold", y=1.02)

    # -- Plot 1: Predictions (Normal vs Anomaly) --
    ax1 = axes[0]
    normal_mask = sample_preds == 1
    anomaly_mask = sample_preds == -1

    ax1.scatter(
        sample_df.loc[sample_df.index[normal_mask], "PacketSize"],
        sample_df.loc[sample_df.index[normal_mask], "Duration"],
        c="#2ecc71", s=8, alpha=0.4, label=f"Normal ({normal_mask.sum():,})"
    )
    ax1.scatter(
        sample_df.loc[sample_df.index[anomaly_mask], "PacketSize"],
        sample_df.loc[sample_df.index[anomaly_mask], "Duration"],
        c="#e74c3c", s=12, alpha=0.7, label=f"Anomaly ({anomaly_mask.sum():,})"
    )
    ax1.set_xlabel("Packet Size (BytesSent + BytesReceived)", fontsize=11)
    ax1.set_ylabel("Duration", fontsize=11)
    ax1.set_title("Model Predictions", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")

    # -- Plot 2: Anomaly Scores Heatmap --
    ax2 = axes[1]
    scatter = ax2.scatter(
        sample_df["PacketSize"], sample_df["Duration"],
        c=sample_scores, cmap="RdYlGn", s=8, alpha=0.6
    )
    ax2.set_xlabel("Packet Size (BytesSent + BytesReceived)", fontsize=11)
    ax2.set_ylabel("Duration", fontsize=11)
    ax2.set_title("Anomaly Scores (green=normal, red=anomalous)", fontsize=13,
                  fontweight="bold")
    plt.colorbar(scatter, ax=ax2, label="Anomaly Score")

    plt.tight_layout()
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  | Chart saved to: {os.path.basename(chart_path)}")
    print(f"  | Sample size:    {sample_size:,} points plotted")
    print(f"  -----------------------------------------------------------")


def run():
    """Execute the full anomaly detection pipeline."""
    print_header()
    start = time.time()

    print("\n  [Step 1/5] Loading processed data...")
    df = load_data(DB_PATH)

    print("\n  [Step 2/5] Training Isolation Forest...")
    model, predictions, scores = train_isolation_forest(df)

    print("\n  [Step 3/5] Evaluating model...")
    evaluate_model(df, predictions)

    print("\n  [Step 4/5] Generating alerts...")
    n_alerts = generate_alerts(df, predictions, scores, DB_PATH)

    print("\n  [Step 5/5] Creating visualization...")
    create_visualization(df, predictions, scores, CHART_PATH)

    elapsed = time.time() - start

    print(f"\n  [DONE] ANOMALY DETECTION COMPLETE in {elapsed:.1f}s")
    print(f"  | Alerts generated:  {n_alerts:>10,}")
    print(f"  | Chart:             {CHART_PATH}")
    print(f"  | Alerts table:      alerts (in network_traffic.db)")

    return n_alerts


if __name__ == "__main__":
    run()
