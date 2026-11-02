"""US airfare forecasting app (the models from M2_arimax.ipynb).

Run from this folder with:   streamlit run flight_app.py
Needs the five FRED files in ML06-second-review/data: CUSR0000SETG01.csv, DJFUELUSGULF.csv,
CPIAUCSL.csv, UNRATE.csv, UMCSENT.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "ML06-second-review", "data")
FILES = {
    "airfare":   "CUSR0000SETG01.csv",   # target
    "jet_fuel":  "DJFUELUSGULF.csv",
    "cpi":       "CPIAUCSL.csv",
    "unemp":     "UNRATE.csv",
    "sentiment": "UMCSENT.csv",
}
LABELS = {
    "airfare": "Airline fare index (CPI)",
    "jet_fuel": "Jet fuel ($ per gallon)",
    "cpi": "Overall prices (CPI)",
    "unemp": "Unemployment rate (%)",
    "sentiment": "Consumer sentiment",
}
FEATURES = ["fuel", "cpi", "unemp", "sentiment", "covid"]
ORDER = (1, 1, 1)
TEST_MONTHS = 36
HORIZONS = [1, 3, 6, 12]
LITRES_PER_GALLON = 3.785


def inr(x, decimals=0):
    """Format a number as rupees with Indian digit grouping: 1234567 -> ₹12,34,567."""
    sign = "-" if x < 0 else ""
    whole, _, frac = f"{abs(x):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, groups = whole[:-3], [whole[-3:]]
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join(([head] if head else []) + groups)
    return f"{sign}₹{whole}" + (f".{frac}" if frac else "")


# ---------------------------------------------------------------------------
# Data and models (same steps as the notebook)
# ---------------------------------------------------------------------------
def load_fred(path, name):
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.set_index(date_col)[value_col].rename(name)
    return s.resample("MS").mean()          # daily -> monthly average


@st.cache_data(show_spinner=False)
def load_data():
    series = {name: load_fred(os.path.join(DATA_DIR, f), name) for name, f in FILES.items()}
    data = pd.concat(series.values(), axis=1)
    data = data.loc[data.dropna().index.min():]
    data = data.interpolate(limit=3, limit_area="inside").dropna()
    data.index.name = "date"
    yoy = data.pct_change(12).dropna() * 100
    lag_corr = [yoy["airfare"].corr(yoy["jet_fuel"].shift(k)) for k in range(13)]
    fuel_lag = max(1, int(np.argmax(lag_corr)))
    return data, yoy, lag_corr, fuel_lag


def make_features(raw, fuel_lag, h=1):
    """Model inputs, each lagged so it is already known h months before the month forecast."""
    d = pd.DataFrame(index=raw.index)
    d["y"] = np.log(raw["airfare"])
    d["fuel"] = np.log(raw["jet_fuel"]).shift(max(fuel_lag, h))
    d["cpi"] = np.log(raw["cpi"]).shift(max(1, h))
    d["unemp"] = raw["unemp"].shift(max(1, h))
    d["sentiment"] = raw["sentiment"].shift(max(1, h))
    d["covid"] = ((d.index >= "2020-03-01") & (d.index <= "2021-06-01")).astype(int)
    return d


def mape(actual, pred):
    return mean_absolute_percentage_error(actual, pred) * 100


def h_step_forecast(d, cols, h, test_start):
    """Forecast each test month from an origin h months earlier (parameters learned on training only)."""
    ex = lambda x: x[cols] if cols else None
    tr = d[d.index < test_start]
    res = SARIMAX(tr["y"], exog=ex(tr), order=ORDER).fit(disp=False, maxiter=1000)
    preds = {}
    for target in d.index[d.index >= test_start]:
        origin = d.index.get_loc(target) - h
        hist = d.iloc[:origin + 1]
        fut = d.iloc[origin + 1: origin + 1 + h]
        m = SARIMAX(hist["y"], exog=ex(hist), order=ORDER).filter(res.params)
        preds[target] = np.exp(m.forecast(steps=h, exog=ex(fut)).iloc[-1])
    return pd.Series(preds)


@st.cache_resource(show_spinner=False)
def backtest():
    """Test on the last 36 months, exactly as in the notebook."""
    data, _, _, fuel_lag = load_data()
    df = make_features(data, fuel_lag).dropna().asfreq("MS")
    test_start = df.index[-TEST_MONTHS]
    train, test = df[df.index < test_start], df[df.index >= test_start]
    actual = data["airfare"].loc[test.index]

    preds = {"Naive (same as last month)": data["airfare"].shift(1).loc[test.index]}
    fits = {}
    for name, cols in [("ARIMA (fare history only)", []), ("ARIMAX (fare history + economy)", FEATURES)]:
        res = SARIMAX(train["y"], exog=train[cols] if cols else None, order=ORDER).fit(disp=False, maxiter=1000)
        full = SARIMAX(df["y"], exog=df[cols] if cols else None, order=ORDER).filter(res.params)
        preds[name] = np.exp(full.get_prediction(start=test.index[0]).predicted_mean)
        fits[name] = res

    metrics = pd.DataFrame({
        name: {"MAPE %": mape(actual, p),
               "RMSE": np.sqrt(mean_squared_error(actual, p)),
               "R²": r2_score(actual, p)}
        for name, p in preds.items()}).T.sort_values("MAPE %")

    rows = []
    for h in HORIZONS:
        d = make_features(data, fuel_lag, h).dropna().asfreq("MS")
        targets = d.index[d.index >= test_start]
        act = data["airfare"].loc[targets]
        rows.append({"Months ahead": h,
                     "Naive": mape(act, data["airfare"].shift(h).loc[targets]),
                     "ARIMA": mape(act, h_step_forecast(d, [], h, test_start)),
                     "ARIMAX": mape(act, h_step_forecast(d, FEATURES, h, test_start))})
    horizon = pd.DataFrame(rows).set_index("Months ahead")

    arimax = fits["ARIMAX (fare history + economy)"]
    coefs = pd.DataFrame({"Coefficient": arimax.params, "p-value": arimax.pvalues})
    return {"test_start": test_start, "actual": actual, "preds": preds, "metrics": metrics,
            "horizon": horizon, "coefs": coefs}


@st.cache_resource(show_spinner=False)
def fit_full(use_macro):
    """Final model: learned on all months, used for forecasting the future."""
    data, _, _, fuel_lag = load_data()
    df = make_features(data, fuel_lag).dropna().asfreq("MS")
    cols = FEATURES if use_macro else []
    return SARIMAX(df["y"], exog=df[cols] if cols else None, order=ORDER).fit(disp=False, maxiter=1000)


def naive_forecast(months):
    """Every future month = the latest month. The range comes from typical month-to-month moves."""
    data = load_data()[0]
    future = pd.date_range(data.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    sd = np.log(data["airfare"]).diff().std()          # typical monthly change, in log terms
    spread = 1.96 * sd * np.sqrt(np.arange(1, months + 1))
    last = data["airfare"].iloc[-1]
    return pd.DataFrame({"forecast": last, "low": last * np.exp(-spread), "high": last * np.exp(spread)},
                        index=future)


def forecast(use_macro, months, fuel_change, inflation, unemp, sentiment):
    """Forecast the next months under a scenario for the economy."""
    data, _, _, fuel_lag = load_data()
    res = fit_full(use_macro)
    future = pd.date_range(data.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    ext = data.reindex(data.index.append(future))
    last = data.iloc[-1]
    ext.loc[future, "jet_fuel"] = last["jet_fuel"] * (1 + fuel_change / 100)
    ext.loc[future, "cpi"] = last["cpi"] * (1 + inflation / 100) ** (np.arange(1, months + 1) / 12)
    ext.loc[future, "unemp"] = unemp
    ext.loc[future, "sentiment"] = sentiment
    exog = make_features(ext, fuel_lag).loc[future, FEATURES] if use_macro else None
    fc = res.get_forecast(steps=months, exog=exog)
    ci = np.exp(fc.conf_int(alpha=0.05))
    return pd.DataFrame({"forecast": np.exp(fc.predicted_mean).to_numpy(),
                         "low": ci.iloc[:, 0].to_numpy(), "high": ci.iloc[:, 1].to_numpy()}, index=future)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Airfare Forecast", page_icon="✈️", layout="wide")
st.title("✈️ US airfare forecast")
st.caption("Forecasts the US airline fare price index (FRED: CUSR0000SETG01), a national monthly average. "
           "It is not the price of a single flight. Rupee amounts assume your ticket moves by the same % "
           "as US fares. A study project.")

missing = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing:
    st.error("Missing data files in ML06-second-review/data: " + ", ".join(missing))
    st.stop()

data, yoy, lag_corr, fuel_lag = load_data()
last = data.iloc[-1]
last_month = data.index[-1]
inflation_12m = (data["cpi"].iloc[-1] / data["cpi"].iloc[-13] - 1) * 100

tab_fc, tab_acc, tab_data = st.tabs(["Forecast", "Model accuracy", "Data and drivers"])

# ---------------------------------------------------------------------------
# Tab 1: forecast
# ---------------------------------------------------------------------------
with tab_fc:
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.subheader("Settings")
        model = st.radio("Model", ["ARIMAX (fare history + economy)", "ARIMA (fare history only)",
                                   "Naive (same as last month)"],
                         help="Test error, 1 month ahead: ARIMAX 1.77%, ARIMA 1.85%, Naive 2.15%. "
                              "Naive is the simple benchmark the other models must beat.")
        use_macro = model.startswith("ARIMAX")
        use_naive = model.startswith("Naive")
        months = st.slider("Months to forecast", 1, 12, 6)

        st.subheader("In Indian rupees")
        price = st.number_input("Your ticket price today (₹)", min_value=100, value=5000, step=500,
                                help="Any domestic or international fare. It is moved by the same % "
                                     "as the forecast fare index.")
        rate = st.number_input("Exchange rate (₹ per US$)", min_value=50.0, max_value=150.0, value=88.0,
                               step=0.5, help="Used to show jet fuel in ₹. Change it to today's rate.")
        fuel_inr = last["jet_fuel"] * rate / LITRES_PER_GALLON

        st.subheader("What if the economy changes?")
        st.caption("ARIMAX only. Each value is held from the next month on. "
                   "These are US figures, because the model was trained on US data.")
        fuel_change = st.slider(f"Jet fuel vs now (now {inr(fuel_inr, 1)} per litre), %", -50, 100, 0, 5,
                                disabled=not use_macro,
                                help=f"US Gulf Coast jet fuel, ${last['jet_fuel']:.2f} per gallon, "
                                     f"converted at ₹{rate:g} per US$.")
        inflation = st.slider("Inflation, % per year", -2.0, 10.0, float(round(inflation_12m, 1)), 0.1,
                              disabled=not use_macro,
                              help=f"Default = the last 12 months ({inflation_12m:.1f}%).")
        unemp = st.slider("Unemployment rate, %", 3.0, 15.0, float(last["unemp"]), 0.1, disabled=not use_macro)
        sentiment = st.slider("Consumer sentiment", 40.0, 110.0, float(last["sentiment"]), 0.5,
                              disabled=not use_macro)

    with st.spinner("Fitting the model on all months..."):
        if use_naive:
            fc = naive_forecast(months)
        else:
            fc = forecast(use_macro, months, fuel_change, inflation, unemp, sentiment)

    with right:
        end = fc.iloc[-1]
        change = (end["forecast"] / last["airfare"] - 1) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Index now ({last_month:%b %Y})", f"{last['airfare']:.1f}")
        m2.metric(f"Forecast for {fc.index[-1]:%b %Y}", f"{end['forecast']:.1f}", f"{change:+.1f}%",
                  delta_color="inverse")
        diff = price * (end["forecast"] / last["airfare"] - 1)
        m3.metric(f"Your {inr(price)} ticket then", inr(price * end["forecast"] / last["airfare"]),
                  ("+" if diff >= 0 else "") + inr(diff), delta_color="inverse")

        hist = data["airfare"].loc["2018":]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Actual", line=dict(color="#555", width=2)))
        band_x = [last_month, *fc.index, *fc.index[::-1], last_month]
        band_y = [last["airfare"], *fc["high"], *fc["low"][::-1], last["airfare"]]
        fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor="rgba(31,119,180,0.15)",
                                 mode="lines", line=dict(width=0), name="95% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[last_month, *fc.index], y=[last["airfare"], *fc["forecast"]],
                                 name="Forecast", line=dict(color="#1f77b4", width=2.5, dash="dash"),
                                 mode="lines+markers"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                          title="Airline fare index: actual and forecast",
                          yaxis_title="Index (1982-84 = 100)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame({
            "Month": fc.index.strftime("%b %Y"),
            "Forecast index": fc["forecast"].round(1),
            "Likely range (95%)": [f"{lo:.1f} to {hi:.1f}" for lo, hi in zip(fc["low"], fc["high"])],
            "Change vs now": [f"{(v / last['airfare'] - 1) * 100:+.1f}%" for v in fc["forecast"]],
            "Your ticket (₹)": [inr(price * v / last["airfare"]) for v in fc["forecast"]],
            "Likely ticket range (₹)": [f"{inr(price * lo / last['airfare'])} to {inr(price * hi / last['airfare'])}"
                                        for lo, hi in zip(fc["low"], fc["high"])],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

        if use_macro:
            params = fit_full(True).params
            st.info(f"**How much the economy moves the forecast.** In this model, 10% more expensive jet fuel "
                    f"changes fares by only about {params['fuel'] * 10:+.2f}% (fuel acts {fuel_lag} months later), "
                    f"about {inr(price * params['fuel'] * 0.10)} on your {inr(price)} ticket. "
                    f"1% higher overall prices change fares by about {params['cpi']:+.2f}%. "
                    f"Most of the forecast comes from the fare's own recent history.")
        st.caption("The range widens with each month ahead: the further out, the less certain the forecast.")

# ---------------------------------------------------------------------------
# Tab 2: model accuracy
# ---------------------------------------------------------------------------
with tab_acc:
    with st.spinner("Testing the models on the last 36 months (first time only)..."):
        bt = backtest()
    ts = bt["test_start"]
    st.write(f"Each model learned from {data.index[0]:%b %Y} to {ts - pd.offsets.MonthBegin(1):%b %Y}, "
             f"then forecast the {TEST_MONTHS} test months from {ts:%b %Y} to {last_month:%b %Y}, "
             "one month ahead at a time.")

    met = bt["metrics"].copy()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(met.style.format({"MAPE %": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"}),
                     use_container_width=True)
        best, naive = met["MAPE %"].iloc[0], met.loc["Naive (same as last month)", "MAPE %"]
        st.caption(f"MAPE = average error in %. The best model is off by {best:.2f}% on average, "
                   f"against {naive:.2f}% for simply repeating last month.")
    with c2:
        st.markdown("**ARIMAX: which inputs matter** (learned on the training months)")
        coefs = bt["coefs"].loc[FEATURES].copy()
        coefs.index = ["Jet fuel", "Overall prices (CPI)", "Unemployment", "Consumer sentiment", "COVID period"]
        coefs["Matters?"] = np.where(coefs["p-value"] < 0.05, "Yes", "No")
        st.dataframe(coefs.style.format({"Coefficient": "{:.4f}", "p-value": "{:.3f}"}), use_container_width=True)
        st.caption("p-value below 0.05 = a reliable effect. Jet fuel is not reliable once prices are included.")

    fig = go.Figure()
    act = data["airfare"].loc["2019":]
    fig.add_trace(go.Scatter(x=act.index, y=act, name="Actual", line=dict(color="#555", width=2)))
    colors = {"Naive (same as last month)": "#bbbbbb", "ARIMA (fare history only)": "#ff7f0e",
              "ARIMAX (fare history + economy)": "#1f77b4"}
    for name, p in bt["preds"].items():
        fig.add_trace(go.Scatter(x=p.index, y=p, name=name, line=dict(color=colors[name], dash="dash")))
    fig.add_vline(x=ts, line_dash="dot", line_color="grey")
    fig.add_annotation(x=ts, y=1, yref="paper", text="test starts", showarrow=False, xanchor="left")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      title="One-month-ahead forecasts on the test months", yaxis_title="Index",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Error grows the further ahead you forecast**")
    hz = bt["horizon"]
    c3, c4 = st.columns([1, 1.4])
    with c3:
        st.dataframe(hz.style.format("{:.2f}%"), use_container_width=True)
    with c4:
        fig = go.Figure()
        for col, color in [("Naive", "#bbbbbb"), ("ARIMA", "#ff7f0e"), ("ARIMAX", "#1f77b4")]:
            fig.add_trace(go.Scatter(x=hz.index, y=hz[col], name=col, mode="lines+markers",
                                     line=dict(color=color)))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Months ahead", tickvals=HORIZONS), yaxis_title="MAPE %")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: data and drivers
# ---------------------------------------------------------------------------
with tab_data:
    st.write(f"{len(data)} months of data from FRED, {data.index[0]:%b %Y} to {last_month:%b %Y}. "
             "Daily jet fuel prices are averaged per month.")
    var = st.selectbox("Compare airfares with", ["jet_fuel", "cpi", "unemp", "sentiment"],
                       format_func=lambda c: LABELS[c])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["airfare"], name=LABELS["airfare"], line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=data.index, y=data[var], name=LABELS[var], yaxis="y2",
                             line=dict(color="#ff7f0e")))
    for a, b in [("2008-01-01", "2009-06-01"), ("2020-02-01", "2020-06-01")]:
        fig.add_vrect(x0=a, x1=b, fillcolor="grey", opacity=0.15, line_width=0)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      title=f"Airfares vs {LABELS[var].lower()} (shaded: 2008 crisis, COVID)",
                      yaxis=dict(title=LABELS["airfare"]),
                      yaxis2=dict(title=LABELS[var], overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        corr = yoy.corr()
        names = [LABELS[c].split(" (")[0] for c in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=corr.round(2).values, texttemplate="%{text}"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title="Correlation of yearly % changes")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=list(range(13)), y=lag_corr,
                               marker_color=["#1f77b4" if k == fuel_lag else "#bbbbbb" for k in range(13)]))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title=f"Jet fuel leads fares by about {fuel_lag} months",
                          xaxis_title="Fuel lag (months)", yaxis_title="Correlation with fare changes")
        st.plotly_chart(fig, use_container_width=True)
