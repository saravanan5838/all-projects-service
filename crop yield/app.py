# Crop Yield Predictor - Streamlit frontend (basic + optimised models)
# Run from the folder that contains app.py and the app_assets folder:
#     python -m streamlit run app.py
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = "app_assets"
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")

CAT_COLS = ["weather_zone", "irrigation_method", "seed_quality",
            "soil_type", "fertilized", "season", "crop_type"]

GROUPS = {
    "Climate": ["avg_temp_c", "rainfall_mm", "humidity_pct", "weather_zone", "season",
                "rain_x_temp", "temp_sq", "rain_per_humidity", "water_stress",
                "total_water_mm"],
    "Soil": ["soil_ph", "soil_type", "nitrogen_kg_ha", "phosphorus_kg_ha",
             "potassium_kg_ha", "npk_total", "n_to_p", "n_to_k", "p_to_k",
             "ph_deviation", "soil_ph_missing", "phosphorus_kg_ha_missing",
             "potassium_kg_ha_missing"],
    "Management & pests": ["fertilized", "irrigation_mm", "irrigation_method",
                           "seed_quality", "days_from_last_harvest", "pest_index"],
    "Crop type": ["crop_type"],
}


# Must be identical to the feature engineering used when the optimised model was trained
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    eps = 1e-6
    for c in ["soil_ph", "phosphorus_kg_ha", "potassium_kg_ha"]:
        d[f"{c}_missing"] = d[c].isna().astype(int)
    d["npk_total"] = d[["nitrogen_kg_ha", "phosphorus_kg_ha",
                        "potassium_kg_ha"]].sum(axis=1, min_count=1)
    d["n_to_p"] = d["nitrogen_kg_ha"] / (d["phosphorus_kg_ha"] + eps)
    d["n_to_k"] = d["nitrogen_kg_ha"] / (d["potassium_kg_ha"] + eps)
    d["p_to_k"] = d["phosphorus_kg_ha"] / (d["potassium_kg_ha"] + eps)
    d["ph_deviation"] = (d["soil_ph"] - 6.5).abs()
    d["total_water_mm"] = d["rainfall_mm"] + d["irrigation_mm"]
    d["rain_x_temp"] = d["rainfall_mm"] * d["avg_temp_c"]
    d["temp_sq"] = d["avg_temp_c"] ** 2
    d["rain_per_humidity"] = d["rainfall_mm"] / (d["humidity_pct"] + eps)
    d["water_stress"] = d["avg_temp_c"] / (d["total_water_mm"] + eps)
    for c in CAT_COLS:
        d[c] = d[c].astype("category")
    return d


@st.cache_resource
def load_assets():
    with open(f"{ASSETS}/meta.json") as f:
        meta = json.load(f)
    models = {}
    if os.path.exists(f"{ASSETS}/linear_basic.joblib"):
        models["Basic - Linear Regression"] = joblib.load(f"{ASSETS}/linear_basic.joblib")
    if os.path.exists(f"{ASSETS}/rf_basic.joblib"):
        models["Basic - Random Forest"] = joblib.load(f"{ASSETS}/rf_basic.joblib")
    models["Optimised - LightGBM"] = joblib.load(f"{ASSETS}/lgbm_optimised.joblib")
    comparison = pd.read_csv(f"{ASSETS}/comparison.csv", index_col=0)
    return meta, models, comparison


meta, models, comparison = load_assets()
OPT = "Optimised - LightGBM"


def predict(name, raw_df):
    """raw_df has the 17 original columns. Returns an array of predictions."""
    X = add_features(raw_df) if name == OPT else raw_df
    return models[name].predict(X)


# ------------------------------------------------------------------ sidebar inputs
st.sidebar.title("Field conditions")
vals = {}


def cat_input(col, label):
    opts = meta["cat_options"][col]
    vals[col] = st.selectbox(label, opts, index=opts.index(meta["cat_defaults"][col]))


def num_input(col, label, step, can_be_missing=False):
    r = meta["num_ranges"][col]
    lo, hi = round(r["min"] / step) * step, round(r["max"] / step) * step
    default = min(max(round(r["default"] / step) * step, lo), hi)
    if can_be_missing and st.checkbox(f"{label}: not available", key=col + "_na"):
        vals[col] = np.nan
        return
    vals[col] = float(st.slider(label, float(lo), float(hi), float(default), float(step)))


with st.sidebar.expander("Crop", expanded=True):
    cat_input("crop_type", "Crop")
with st.sidebar.expander("Climate", expanded=True):
    num_input("avg_temp_c", "Average temperature (C)", 1.0)
    num_input("rainfall_mm", "Rainfall (mm)", 10.0)
    num_input("humidity_pct", "Humidity (%)", 0.5)
    cat_input("weather_zone", "Weather zone")
    cat_input("season", "Season")
with st.sidebar.expander("Soil", expanded=False):
    cat_input("soil_type", "Soil type")
    num_input("soil_ph", "Soil pH", 0.05, can_be_missing=True)
    num_input("nitrogen_kg_ha", "Nitrogen (kg/ha)", 1.0)
    num_input("phosphorus_kg_ha", "Phosphorus (kg/ha)", 1.0, can_be_missing=True)
    num_input("potassium_kg_ha", "Potassium (kg/ha)", 1.0, can_be_missing=True)
with st.sidebar.expander("Management and pests", expanded=False):
    cat_input("fertilized", "Fertilised")
    cat_input("seed_quality", "Seed quality")
    cat_input("irrigation_method", "Irrigation method")
    num_input("irrigation_mm", "Irrigation (mm)", 1.0)
    num_input("pest_index", "Pest index", 1.0)
    num_input("days_from_last_harvest", "Days since last harvest", 1.0)

row = pd.DataFrame([vals])[meta["columns"]]

# ------------------------------------------------------------------ main page
st.title("Explainable Crop Yield Prediction")
st.caption("Climate and soil variables integrated in one model. The training data appears "
           "to be synthetic, so use this as a demonstration, not for real farming decisions.")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explanation", "What-if", "Model comparison"])

# ---- Tab 1: prediction
with tab1:
    choice = st.selectbox("Choose model", list(models), index=list(models).index(OPT))
    st.metric(f"Predicted yield ({choice})", f"{predict(choice, row)[0]:.2f} tons")

    all_preds = pd.DataFrame({
        "Model": list(models),
        "Predicted yield (tons)": [float(predict(m, row)[0]) for m in models],
    }).set_index("Model")
    all_preds["Difference vs optimised"] = all_preds["Predicted yield (tons)"] - \
        all_preds.loc[OPT, "Predicted yield (tons)"]
    st.subheader("Same inputs, every model")
    st.dataframe(all_preds.round(2), use_container_width=True)
    st.caption("The basic model gives only a number. The optimised model also explains "
               "its prediction (see the Explanation tab).")

# ---- Tab 2: explanation (optimised model, exact SHAP values)
with tab2:
    st.write("Explanations come from the optimised LightGBM model (exact SHAP values). "
             "Green bars raise the predicted yield, orange bars lower it.")
    X_fe = add_features(row)
    c = models[OPT].predict(X_fe, pred_contrib=True)[0]
    contrib, base = pd.Series(c[:-1], index=X_fe.columns), c[-1]
    pred = base + contrib.sum()

    col1, col2 = st.columns(2)
    with col1:
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])
        labels = [f"{f} = {X_fe.iloc[0][f]:.3g}" if f not in CAT_COLS
                  else f"{f} = {X_fe.iloc[0][f]}" for f in top.index]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(labels[::-1], top.values[::-1],
                color=np.where(top.values[::-1] >= 0, "#1b9e77", "#d95f02"))
        ax.axvline(0, color="gray")
        ax.set_title(f"Predicted {pred:.2f} t (average field {base:.2f} t)", fontsize=10)
        ax.set_xlabel("SHAP value (tons)")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        grp = pd.Series({g: contrib[[f for f in fs if f in contrib.index]].sum()
                         for g, fs in GROUPS.items()})
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(grp.index[::-1], grp.values[::-1],
                color=np.where(grp.values[::-1] >= 0, "#1b9e77", "#d95f02"))
        ax.axvline(0, color="gray")
        ax.set_title("Net effect of each variable group", fontsize=10)
        ax.set_xlabel("SHAP value (tons)")
        plt.tight_layout()
        st.pyplot(fig)

# ---- Tab 3: what-if curves
with tab3:
    var = st.radio("Vary", ["avg_temp_c", "rainfall_mm"], horizontal=True)
    crops_sel = st.multiselect("Crops", meta["cat_options"]["crop_type"],
                               default=[vals["crop_type"]])
    default_models = [m for m in models if m in (OPT, "Basic - Linear Regression")]
    models_sel = st.multiselect("Models", list(models), default=default_models)

    r = meta["num_ranges"][var]
    grid = np.linspace(r["min"], r["max"], 30)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, crop in enumerate(crops_sel):
        sweep = pd.DataFrame([{**vals, "crop_type": crop, var: g} for g in grid])
        sweep = sweep[meta["columns"]]
        for m in models_sel:
            ax.plot(grid, predict(m, sweep), color=plt.cm.tab10(i),
                    ls="-" if m == OPT else "--", lw=2,
                    label=f"{crop} ({m.split(' - ')[0].lower()})")
    ax.set_xlabel(var)
    ax.set_ylabel("Predicted yield (tons)")
    ax.set_title("What-if: other inputs stay as set in the sidebar")
    if crops_sel and models_sel:
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Solid = optimised, dashed = basic. Curves stay inside the range seen in "
               "the training data because tree models cannot extrapolate beyond it.")

# ---- Tab 4: comparison table
with tab4:
    st.subheader("Basic vs optimised (test set, 49,464 rows)")
    st.dataframe(comparison, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Test R2 (higher is better)")
        st.bar_chart(comparison["Test R2"])
    with c2:
        st.write("Test RMSE in tons (lower is better)")
        st.bar_chart(comparison["RMSE"])