"""
Smart Grid Power Quality Anomaly Detection - Streamlit frontend
Run:  streamlit run app.py

Files needed in the SAME folder as this app:
    pq_features.py            (shared feature code)
    optimised_model.joblib    (saved from the notebook)
    comparison_results.csv    (from optimised_isolation_forest.py)
    per_disturbance_results.csv
    Test_Data.csv             (optional, for the 'use sample data' option)
"""

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

import pq_features as pq

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Smart Grid PQ Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    path = BASE_DIR / "optimised_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner="Building features and scoring data ...")
def score_file(file_bytes: bytes):
    bundle = load_bundle()
    df = pq.load_pq_csv(io.BytesIO(file_bytes))
    missing = pq.missing_columns(df)
    if missing:
        return df, None, missing
    feats = pq.build_features(df)[bundle["feature_set"]][bundle["columns"]]
    scores = -bundle["model"].score_samples(bundle["scaler"].transform(feats))
    return df, scores, []


# ----------------------------------------------------------------------
# Page 1: Detection
# ----------------------------------------------------------------------
def page_detect():
    st.header("Detect power quality anomalies")
    bundle = load_bundle()
    if bundle is None:
        st.error("`optimised_model.joblib` was not found in the app folder. "
                 "Save the model from your notebook first (see the setup steps).")
        return

    source = st.radio(
        "Data source",
        ["Upload CSV", "Use Test_Data.csv from the project folder"],
        horizontal=True,
    )
    file_bytes = None
    if source == "Upload CSV":
        up = st.file_uploader("Upload a CSV in the same format as the training data",
                              type=["csv"])
        if up is not None:
            file_bytes = up.getvalue()
    else:
        sample = BASE_DIR / "Test_Data.csv"
        if sample.exists():
            file_bytes = sample.read_bytes()
        else:
            st.warning("Test_Data.csv is not in the app folder.")

    if file_bytes is None:
        st.info("Waiting for data ...")
        return

    df, scores, missing = score_file(file_bytes)
    if missing:
        st.error("The file is missing required columns: " + ", ".join(missing))
        return
    if len(df) < 2 * pq.WINDOW:
        st.warning("This file is very short. The rolling features use the previous "
                   f"{pq.WINDOW} minutes, so results may be unreliable.")

    # ---- threshold control ------------------------------------------
    saved_thr = float(bundle["threshold"])
    lo, hi = float(scores.min()), float(scores.max())
    thr = saved_thr
    if hi > lo:
        thr = st.slider(
            "Alert threshold (higher = fewer alerts, lower = more alerts)",
            min_value=lo, max_value=hi,
            value=float(min(max(saved_thr, lo), hi)),
            step=max((hi - lo) / 500, 1e-6), format="%.4f",
        )
        st.caption(f"Default threshold chosen during training: {saved_thr:.4f}")

    flagged = scores >= thr

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows analysed", f"{len(df):,}")
    c2.metric("Anomalies flagged", f"{int(flagged.sum()):,}")
    c3.metric("Share flagged", f"{100 * flagged.mean():.3f}%")
    if df[pq.TIME_COL].notna().any():
        c4.metric("Period", f"{df[pq.TIME_COL].min():%d %b %Y} - "
                            f"{df[pq.TIME_COL].max():%d %b %Y}")

    # ---- signal plot -------------------------------------------------
    st.subheader("Signal with flagged anomalies")
    plot_col = st.selectbox("Signal to plot", pq.BASE_COLS, index=1)
    if df[pq.TIME_COL].notna().all():
        x = df[pq.TIME_COL].to_numpy()
    else:
        x = np.arange(len(df))
    y = df[plot_col].to_numpy()

    step = max(1, len(df) // 20000)
    idx = np.flatnonzero(flagged)
    if len(idx) > 5000:
        idx = idx[np.linspace(0, len(idx) - 1, 5000).astype(int)]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x[::step], y[::step], lw=0.6, label="Signal")
    ax.scatter(x[idx], y[idx], s=14, c="red", zorder=3, label="Flagged anomaly")
    ax.set_ylabel(plot_col)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    # ---- evaluation when labels exist --------------------------------
    if pq.LABEL_COL in df.columns:
        st.subheader("Evaluation against the labels in this file")
        y_true = (df[pq.LABEL_COL] != 0).to_numpy().astype(int)
        pred = flagged.astype(int)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.3f}")
        m2.metric("Recall", f"{recall_score(y_true, pred, zero_division=0):.3f}")
        m3.metric("F1-score", f"{f1_score(y_true, pred, zero_division=0):.3f}")
        if 0 < y_true.sum() < len(y_true):
            m4.metric("ROC-AUC", f"{roc_auc_score(y_true, scores):.3f}")

        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        tp = int(((y_true == 1) & (pred == 1)).sum())
        st.dataframe(
            pd.DataFrame([[tn, fp], [fn, tp]],
                         index=["Actual normal", "Actual anomaly"],
                         columns=["Predicted normal", "Predicted anomaly"]),
            width="stretch",
        )

        types = sorted(t for t in df[pq.LABEL_COL].unique() if t != 0)
        if types:
            rows = []
            for t in types:
                m = (df[pq.LABEL_COL] == t).to_numpy()
                rows.append({"Disturbance type": t, "Events": int(m.sum()),
                             "Detected": int(flagged[m].sum()),
                             "Recall": round(float(flagged[m].mean()), 3)})
            st.write("Detection by disturbance type")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- alerts table + download ------------------------------------
    st.subheader("Alerts (highest score first)")
    alerts = df.loc[flagged, [pq.TIME_COL] + pq.BASE_COLS].copy()
    alerts.insert(1, "Anomaly score", scores[flagged])
    if pq.LABEL_COL in df.columns:
        alerts[pq.LABEL_COL] = df.loc[flagged, pq.LABEL_COL]
    alerts = alerts.sort_values("Anomaly score", ascending=False)
    st.dataframe(alerts.head(1000), width="stretch", hide_index=True)
    st.download_button(
        "Download all alerts as CSV",
        alerts.to_csv(index=False).encode("utf-8"),
        file_name="anomaly_alerts.csv",
        mime="text/csv",
    )

    with st.expander("Anomaly score distribution"):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(scores, bins=80, color="tab:blue", alpha=0.8)
        ax.axvline(thr, color="red", ls="--", label="Threshold")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Rows")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Page 2: Model comparison
# ----------------------------------------------------------------------
def page_compare():
    st.header("Basic vs Optimised Isolation Forest")
    comp_path = BASE_DIR / "comparison_results.csv"
    type_path = BASE_DIR / "per_disturbance_results.csv"
    if not comp_path.exists():
        st.error("`comparison_results.csv` not found. Run optimised_isolation_forest.py first.")
        return

    comp = pd.read_csv(comp_path).set_index("model")
    st.subheader("Test-set metrics")
    st.dataframe(comp.round(4), width="stretch")

    basic = comp.loc["Basic IsolationForest"]
    opt = comp.loc["Optimised IsolationForest"]
    metrics = ["precision", "recall", "f1", "roc_auc", "avg_precision"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "Avg precision"]
    x = np.arange(len(metrics))
    w = 0.38

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        b1 = ax.bar(x - w / 2, basic[metrics], w, label="Basic")
        b2 = ax.bar(x + w / 2, opt[metrics], w, label="Optimised")
        ax.bar_label(b1, fmt="%.3f", fontsize=7)
        ax.bar_label(b2, fmt="%.3f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Metrics comparison")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        bars = ax.bar(["Basic", "Optimised"],
                      [basic["flagged_%"], opt["flagged_%"]],
                      color=["tab:blue", "tab:orange"])
        ax.bar_label(bars, fmt="%.2f%%")
        ax.set_ylabel("% of rows flagged")
        ax.set_title("Alert volume (lower = fewer false alarms)")
        st.pyplot(fig)
        plt.close(fig)

    if type_path.exists():
        per_type = pd.read_csv(type_path)
        st.subheader("Detection rate by disturbance type")
        st.dataframe(per_type, width="stretch", hide_index=True)
        xt = np.arange(len(per_type))
        fig, ax = plt.subplots(figsize=(8, 4))
        b1 = ax.bar(xt - w / 2, per_type["basic_recall"], w, label="Basic")
        b2 = ax.bar(xt + w / 2, per_type["optimised_recall"], w, label="Optimised")
        ax.bar_label(b1, fmt="%.2f", fontsize=8)
        ax.bar_label(b2, fmt="%.2f", fontsize=8)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"Type {int(t)}\n({int(n)} events)" for t, n in
                            zip(per_type["disturbance_type"], per_type["events"])])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Recall")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    grid_path = BASE_DIR / "grid_search_results.csv"
    if grid_path.exists():
        with st.expander("Top grid-search configurations (validation data)"):
            st.dataframe(pd.read_csv(grid_path).head(10),
                         width="stretch", hide_index=True)


# ----------------------------------------------------------------------
# Page 3: About
# ----------------------------------------------------------------------
def page_about():
    st.header("About this project")
    st.markdown(
        """
**Goal:** automatically flag abnormal power quality events in smart grid
voltage and THD measurements, without needing labels at detection time.

**Method:** Isolation Forest, an unsupervised algorithm that isolates rare,
unusual points with fewer random splits than normal points.

**Basic model:** default Isolation Forest on 8 raw readings.

**Optimised model:**
- engineered features (ranges, ratios, minute-to-minute change, 30-minute rolling deviation),
- trained on normal data only,
- hyper-parameters tuned with grid search on a time-based validation split,
- alert threshold tuned on validation data.

**Limitations:** some disturbance types (gradual or subtle ones) are hard for
Isolation Forest to detect; a supervised second stage could classify them.
        """
    )


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------
st.title("⚡ Smart Grid Power Quality Anomaly Detection")
page = st.sidebar.radio("Navigate", ["Detect anomalies", "Model comparison", "About"])
if page == "Detect anomalies":
    page_detect()
elif page == "Model comparison":
    page_compare()
else:
    page_about()
