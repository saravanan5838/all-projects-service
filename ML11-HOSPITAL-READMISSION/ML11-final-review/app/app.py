"""Hospital readmission risk app for Model 1 and Model 2.

Run from this folder with:   streamlit run app.py
The models are trained once from diabetic_data.csv, exactly as in
M1_lace_points_score.ipynb (Model 1) and M2_learned_points_score.ipynb (Model 2).
"""
import os
import random
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "ML11-second-review", "data", "diabetic_data.csv")
RESULTS = os.path.join(HERE, "..", "..", "ML11-second-review", "results")
SEED = 42

# ---------------------------------------------------------------------------
# Shared definitions (copied from the notebooks)
# ---------------------------------------------------------------------------
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
            "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
            "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
            "metformin-pioglitazone"]
COUNT_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
DIAG_GROUPS = ["diabetes", "circulatory", "respiratory", "digestive", "injury",
               "musculoskeletal", "genitourinary", "neoplasms"]
DISCHARGE_GROUPS = ["home", "home_health", "facility"]


def diagnosis_group(code):
    """Map an ICD-9 diagnosis code to a clinical group."""
    if pd.isna(code):
        return "other"
    code = str(code)
    if code[0] in "VE":
        return "other"
    try:
        c = float(code)
    except ValueError:
        return "other"
    if 250 <= c < 251: return "diabetes"
    if 390 <= c <= 459 or c == 785: return "circulatory"
    if 460 <= c <= 519 or c == 786: return "respiratory"
    if 520 <= c <= 579 or c == 787: return "digestive"
    if 800 <= c <= 999: return "injury"
    if 710 <= c <= 739: return "musculoskeletal"
    if 580 <= c <= 629 or c == 788: return "genitourinary"
    if 140 <= c <= 239: return "neoplasms"
    return "other"


def discharge_group(d):
    if d == 1: return "home"
    if d in (6, 8): return "home_health"
    if d in (2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30): return "facility"
    return "other"


LABELS = {
    "number_inpatient": "Inpatient admissions (past year)",
    "number_emergency": "Emergency visits (past year)",
    "number_outpatient": "Outpatient visits (past year)",
    "number_diagnoses": "Number of diagnoses",
    "num_medications": "Medications during stay",
    "num_lab_procedures": "Lab tests during stay",
    "num_procedures": "Procedures during stay",
    "time_in_hospital": "Length of stay (days)",
    "n_diabetes_meds": "Diabetes medications",
    "n_dose_changes": "Diabetes dose changes",
    "emergency_admission": "Emergency admission",
    "on_diabetes_med": "On diabetes medication",
    "med_change": "Medication changed during stay",
    "insulin": "On insulin",
    "a1c_measured": "HbA1c measured",
    "a1c_high": "HbA1c high",
    "glucose_high": "Glucose high",
    "discharge_home_health": "Discharged with home health care",
    "discharge_facility": "Discharged to a care facility",
    "discharge_home": "Discharged home",
    "age": "Age",
}


def pretty(text):
    """Replace column names in a rule with readable names."""
    for name in sorted(LABELS, key=len, reverse=True):
        text = text.replace(name, LABELS[name])
    return text.replace("diag_", "Main diagnosis: ")


# Suggested actions, written by hand (NOT learned by the model). For clinician review.
ACTIONS = {
    "number_inpatient": "Repeated admissions: arrange a follow-up call within 2-3 days of discharge.",
    "number_emergency": "Frequent emergency visits: give the patient a clear plan for who to contact first.",
    "num_medications": "Many medications: request a pharmacist medication review before discharge.",
    "n_diabetes_meds": "Several diabetes drugs: check the regimen is understood before discharge.",
    "n_dose_changes": "Doses changed this stay: explain the new doses and give a written list.",
    "med_change": "Medication changed this stay: explain the changes and give a written list.",
    "insulin": "On insulin: confirm the patient can dose and monitor at home.",
    "time_in_hospital": "Long stay: confirm the patient is ready for discharge and has support at home.",
    "number_diagnoses": "Many conditions: arrange an early primary care visit.",
    "discharge_facility": "Going to a care facility: send a complete handover to the receiving team.",
    "discharge_home_health": "Home health care: confirm the first home visit is booked.",
    "a1c_high": "High HbA1c: arrange diabetes education or a diabetes clinic follow-up.",
    "glucose_high": "High glucose: review blood sugar control before discharge.",
    "age": "Older patient: check whether help is needed at home.",
}


def build_features(df):
    """The 29 pieces of information Model 2 uses (Model 1 uses 4 of them)."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
    for c in COUNT_COLS:
        X[c] = df[c].astype(float)
    meds = df[[c for c in MED_COLS if c in df.columns]].astype(str)
    X["n_diabetes_meds"] = (meds != "No").sum(axis=1).astype(float)
    X["n_dose_changes"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
    X["emergency_admission"] = (df["admission_type_id"] == 1).astype(int)
    X["insulin"] = (df["insulin"].astype(str) != "No").astype(int)
    X["med_change"] = (df["change"].astype(str) == "Ch").astype(int)
    X["on_diabetes_med"] = (df["diabetesMed"].astype(str) == "Yes").astype(int)
    a1c = df["A1Cresult"].fillna("None").astype(str)
    X["a1c_measured"] = (a1c != "None").astype(int)
    X["a1c_high"] = a1c.isin([">7", ">8"]).astype(int)
    X["glucose_high"] = df["max_glu_serum"].fillna("None").astype(str).isin([">200", ">300"]).astype(int)
    disch = df["discharge_disposition_id"].map(discharge_group)
    for g in DISCHARGE_GROUPS:
        X[f"discharge_{g}"] = (disch == g).astype(int)
    diag = df["diag_1"].map(diagnosis_group)
    for g in DIAG_GROUPS:
        X[f"diag_{g}"] = (diag == g).astype(int)
    return X


# ---- Model 1: clinical points score (LACE-style) ----
def model1_parts(X):
    """Points for each of the four parts of the Model 1 score."""
    los = X["time_in_hospital"]
    return pd.DataFrame({
        "Length of stay": np.select([los >= 14, los >= 7, los >= 4], [7, 5, 4], default=los.clip(upper=3)),
        "Emergency admission": 3 * X["emergency_admission"],
        "Number of diagnoses": (X["number_diagnoses"] - 4).clip(lower=0, upper=5),
        "Emergency visits last year": X["number_emergency"].clip(upper=4),
    }, index=X.index).astype(int)


MODEL1_TABLE = pd.DataFrame({
    "Risk factor": ["Length of stay", "Emergency admission", "Number of diagnoses", "Emergency visits last year"],
    "How points are given": ["1 day = 1, 2 days = 2, 3 days = 3, 4-6 days = 4, 7-13 days = 5, 14+ days = 7",
                             "Yes = 3, No = 0",
                             "1 point for each diagnosis above 4 (maximum 5)",
                             "1 point for each visit (maximum 4)"],
    "Maximum": [7, 3, 5, 4],
})


# ---- Model 2: learned points score + IF-THEN rules ----
def to_conditions(X, cuts):
    B = pd.DataFrame(index=X.index)
    for c in X.columns:
        if c in cuts:
            for t in cuts[c]:
                B[f"{c} >= {t:g}"] = (X[c] >= t).astype(int)
        else:
            B[c] = X[c]
    return B


def learn_points(B_fit, y_fit, max_items, max_points=5):
    coef = None
    for C in np.logspace(-4, 0, 25):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(B_fit, y_fit)
        if 0 < np.count_nonzero(m.coef_) <= max_items:
            coef = m.coef_.ravel()
    pts = np.round(coef / np.abs(coef).max() * max_points).astype(int)
    pts = pd.Series(pts, index=B_fit.columns)
    return pts[pts != 0]


def rule_text(steps, yes_no):
    """Turn the tree's steps (column, cut point, above?) into short, non-repeating conditions."""
    low, high, yesno, order = {}, {}, {}, []
    for f, thr, above in steps:
        if f not in order:
            order.append(f)
        if f in yes_no:
            yesno[f] = "Yes" if above else "No"
        elif above:
            low[f] = max(low.get(f, -np.inf), int(np.floor(thr)) + 1)
        else:
            high[f] = min(high.get(f, np.inf), int(np.floor(thr)))
    out = []
    for f in order:
        if f in yesno:
            out.append(f"{f} = {yesno[f]}")
        elif f in low and f in high:
            out.append(f"{f} = {low[f]}" if low[f] == high[f] else f"{f} between {low[f]} and {high[f]}")
        elif f in low:
            out.append(f"{f} >= {low[f]}")
        else:
            out.append(f"{f} = 0" if high[f] == 0 else f"{f} <= {high[f]}")
    return [pretty(c) for c in out]


@st.cache_resource(show_spinner=False)
def train():
    """Train both models once, with the same cleaning and patient split as the notebooks."""
    df = pd.read_csv(DATA, na_values="?", low_memory=False)
    df = df[~df["discharge_disposition_id"].isin({11, 13, 14, 19, 20, 21})]
    df = df[df["gender"].isin(["Male", "Female"])].reset_index(drop=True)
    y = (df["readmitted"] == "<30").astype(int).to_numpy()
    groups = df["patient_nbr"].to_numpy()
    X = build_features(df)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(X, y, groups))
    fit_part, val_part = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
                              .split(X.iloc[tr], y[tr], groups[tr]))
    fit_idx, val_idx = tr[fit_part], tr[val_part]
    Xtr, ytr = X.iloc[tr], y[tr]

    # Model 1
    s1_tr = model1_parts(Xtr).sum(axis=1).to_numpy(float)
    risk1 = LogisticRegression().fit(s1_tr.reshape(-1, 1), ytr)
    cutoff1 = np.ceil(np.quantile(s1_tr, 0.75))

    # Model 2: points score
    yes_no = [c for c in X.columns if set(X[c].unique()) <= {0, 1}]
    numbers = [c for c in X.columns if c not in yes_no]
    cuts = {}
    for c in numbers:
        qs = np.unique(np.quantile(Xtr[c], [0.25, 0.5, 0.75, 0.9]))
        cuts[c] = [q for q in qs if Xtr[c].min() < q <= Xtr[c].max()]
    B = to_conditions(X, cuts)
    best_k, best_auc = None, -1
    for k in [4, 6, 8, 10, 12, 15]:
        pts = learn_points(B.iloc[fit_idx], y[fit_idx], k)
        auc = roc_auc_score(y[val_idx], (B.iloc[val_idx][pts.index] @ pts).to_numpy(float))
        if auc > best_auc:
            best_k, best_auc = k, auc
    points = learn_points(B.iloc[tr], ytr, best_k)
    s2_tr = (B.iloc[tr][points.index] @ points).to_numpy(float)
    risk2 = LogisticRegression().fit(s2_tr.reshape(-1, 1), ytr)
    cutoff2 = np.ceil(np.quantile(s2_tr, 0.75))
    median_risk2 = np.median(risk2.predict_proba(s2_tr.reshape(-1, 1))[:, 1])

    # Model 2: IF-THEN rules
    best_leaves, best_auc = None, -1
    for leaves in [4, 6, 8, 12, 16]:
        t = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=200, random_state=0)
        t.fit(X.iloc[fit_idx], y[fit_idx])
        auc = roc_auc_score(y[val_idx], t.predict_proba(X.iloc[val_idx])[:, 1])
        if auc > best_auc:
            best_leaves, best_auc = leaves, auc
    tree = DecisionTreeClassifier(max_leaf_nodes=best_leaves, min_samples_leaf=200, random_state=0).fit(Xtr, ytr)
    leaf_stats = pd.Series(ytr).groupby(tree.apply(Xtr)).agg(["mean", "size"])

    return {
        "df": df, "X": X, "y": y, "test_idx": te, "columns": list(X.columns), "base_rate": ytr.mean(),
        "risk1": risk1, "cutoff1": cutoff1,
        "cuts": cuts, "points": points, "risk2": risk2, "cutoff2": cutoff2, "median_risk2": median_risk2,
        "score2_range": (int(s2_tr.min()), int(s2_tr.max())),
        "yes_no": yes_no, "tree": tree, "leaf_stats": leaf_stats,
    }


def tree_path(M, xrow):
    """The tree's steps for one patient, plus the leaf it lands in."""
    t, names, node, steps = M["tree"].tree_, M["columns"], 0, []
    while t.children_left[node] != -1:
        f, thr = names[t.feature[node]], t.threshold[node]
        above = xrow[f] > thr
        steps.append((f, thr, above))
        node = t.children_right[node] if above else t.children_left[node]
    return steps, node


def all_rules(M):
    t, names, out = M["tree"].tree_, M["columns"], []
    def walk(node, steps):
        if t.children_left[node] == -1:
            out.append((M["leaf_stats"].loc[node, "mean"], M["leaf_stats"].loc[node, "size"], steps))
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], steps + [(f, thr, False)])
        walk(t.children_right[node], steps + [(f, thr, True)])
    walk(0, [])
    return sorted(out, key=lambda r: -r[0])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="wide")
st.title("🏥 30-day readmission risk")
st.caption("Diabetic inpatients, UCI 130-hospitals dataset. A study project: not for real clinical decisions.")

if not os.path.exists(DATA):
    st.error(f"Could not find diabetic_data.csv in ML11-second-review/data/ ({DATA}).")
    st.stop()

with st.spinner("Training the models on diabetic_data.csv (only the first time, about 30 seconds)..."):
    M = train()

AGE_BANDS = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
DISCHARGE_OPTIONS = {"Home": "home", "Home with home health care": "home_health",
                     "Care facility (nursing home, rehab, other hospital)": "facility", "Other": "other"}
DIAG_OPTIONS = {g.capitalize(): g for g in DIAG_GROUPS} | {"Other": "other"}
A1C_OPTIONS = {"Not measured": "None", "Normal": "Norm", "> 7%": ">7", "> 8%": ">8"}
GLU_OPTIONS = {"Not measured": "None", "Normal": "Norm", "> 200": ">200", "> 300": ">300"}
DEFAULTS = dict(age="[60-70)", los=4, n_lab=43, n_proc=1, n_med=16, n_out=0, n_emer=0, n_inp=0, n_diag=8,
                emerg=True, discharge="Home", diag="Circulatory", a1c="Not measured", glu="Not measured",
                insulin=False, med_change=False, on_dm=True, n_dm=1, n_dose=0)

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("loaded", None)


def clear_loaded():
    st.session_state["loaded"] = None


def reset_form():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    clear_loaded()


def load_test_patient():
    """Fill the form with a random patient from the test set (never used for training)."""
    i = random.choice(list(M["test_idx"]))
    raw, x = M["df"].loc[i], M["X"].loc[i]
    flip = lambda d, v: next(k for k, val in d.items() if val == v)
    st.session_state.update(
        age=raw["age"], los=int(x["time_in_hospital"]), n_lab=int(x["num_lab_procedures"]),
        n_proc=int(x["num_procedures"]), n_med=int(x["num_medications"]), n_out=int(x["number_outpatient"]),
        n_emer=int(x["number_emergency"]), n_inp=int(x["number_inpatient"]), n_diag=int(x["number_diagnoses"]),
        emerg=bool(x["emergency_admission"]),
        discharge=flip(DISCHARGE_OPTIONS, discharge_group(raw["discharge_disposition_id"])),
        diag=flip(DIAG_OPTIONS, diagnosis_group(raw["diag_1"])),
        a1c=flip(A1C_OPTIONS, raw["A1Cresult"] if pd.notna(raw["A1Cresult"]) else "None"),
        glu=flip(GLU_OPTIONS, raw["max_glu_serum"] if pd.notna(raw["max_glu_serum"]) else "None"),
        insulin=bool(x["insulin"]), med_change=bool(x["med_change"]), on_dm=bool(x["on_diabetes_med"]),
        n_dm=int(x["n_diabetes_meds"]), n_dose=int(x["n_dose_changes"]),
    )
    st.session_state["loaded"] = (raw["patient_nbr"], raw["gender"], int(M["y"][i]))


tab_check, tab_how, tab_perf = st.tabs(["Check a patient", "How the models work", "Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: check a patient
# ---------------------------------------------------------------------------
with tab_check:
    left, right = st.columns([1, 2], gap="large")

    with left:
        b1, b2 = st.columns(2)
        b1.button("🎲 Load a test patient", on_click=load_test_patient, use_container_width=True)
        b2.button("↺ Reset", on_click=reset_form, use_container_width=True)

        st.subheader("Patient details")
        kw = {"on_change": clear_loaded}
        st.selectbox("Age group", AGE_BANDS, key="age", **kw)
        st.selectbox("Main diagnosis", list(DIAG_OPTIONS), key="diag", **kw)

        with st.expander("This hospital stay", expanded=True):
            st.toggle("Emergency admission", key="emerg", **kw)
            st.number_input("Length of stay (days)", 1, 14, key="los", **kw)
            st.number_input("Number of diagnoses", 1, 16, key="n_diag", **kw)
            st.number_input("Medications during stay", 1, 81, key="n_med", **kw)
            st.number_input("Lab tests during stay", 1, 132, key="n_lab", **kw)
            st.number_input("Procedures during stay", 0, 6, key="n_proc", **kw)
            st.selectbox("Discharged to", list(DISCHARGE_OPTIONS), key="discharge", **kw)

        with st.expander("Past year", expanded=True):
            st.number_input("Inpatient admissions", 0, 21, key="n_inp", **kw)
            st.number_input("Emergency visits", 0, 76, key="n_emer", **kw)
            st.number_input("Outpatient visits", 0, 42, key="n_out", **kw)

        with st.expander("Diabetes care"):
            st.toggle("On diabetes medication", key="on_dm", **kw)
            st.toggle("On insulin", key="insulin", **kw)
            st.toggle("Medication changed during stay", key="med_change", **kw)
            st.number_input("Number of diabetes medications", 0, 6, key="n_dm", **kw)
            st.number_input("Number of dose changes", 0, 4, key="n_dose", **kw)
            st.selectbox("HbA1c result", list(A1C_OPTIONS), key="a1c", **kw)
            st.selectbox("Max glucose", list(GLU_OPTIONS), key="glu", **kw)

    # Build the patient's feature row from the form
    s = st.session_state
    a1c, glu = A1C_OPTIONS[s.a1c], GLU_OPTIONS[s.glu]
    row = {
        "age": int(s.age[1:].split("-")[0]) + 5,
        "time_in_hospital": s.los, "num_lab_procedures": s.n_lab, "num_procedures": s.n_proc,
        "num_medications": s.n_med, "number_outpatient": s.n_out, "number_emergency": s.n_emer,
        "number_inpatient": s.n_inp, "number_diagnoses": s.n_diag,
        "n_diabetes_meds": s.n_dm, "n_dose_changes": s.n_dose,
        "emergency_admission": int(s.emerg), "insulin": int(s.insulin), "med_change": int(s.med_change),
        "on_diabetes_med": int(s.on_dm), "a1c_measured": int(a1c != "None"),
        "a1c_high": int(a1c in (">7", ">8")), "glucose_high": int(glu in (">200", ">300")),
    }
    for g in DISCHARGE_GROUPS:
        row[f"discharge_{g}"] = int(DISCHARGE_OPTIONS[s.discharge] == g)
    for g in DIAG_GROUPS:
        row[f"diag_{g}"] = int(DIAG_OPTIONS[s.diag] == g)
    x = pd.DataFrame([row])[M["columns"]].astype(float)
    base = M["base_rate"]

    # Model 1
    parts1 = model1_parts(x).iloc[0]
    score1 = int(parts1.sum())
    risk1 = M["risk1"].predict_proba([[score1]])[0, 1]
    high1 = score1 >= M["cutoff1"]

    # Model 2
    b = to_conditions(x, M["cuts"]).iloc[0]
    points = M["points"]
    fired = sorted([(c, int(points[c])) for c in points.index if b[c] == 1], key=lambda r: -r[1])
    score2 = sum(p for _, p in fired)
    risk2 = M["risk2"].predict_proba([[score2]])[0, 1]
    if score2 >= M["cutoff2"]:
        tier2 = ("HIGH RISK", "🔴", "error")
    elif risk2 > M["median_risk2"]:
        tier2 = ("MODERATE RISK", "🟡", "warning")
    else:
        tier2 = ("LOW RISK", "🟢", "success")
    steps, leaf = tree_path(M, x.iloc[0])
    leaf_risk, leaf_n = M["leaf_stats"].loc[leaf, "mean"], M["leaf_stats"].loc[leaf, "size"]

    with right:
        loaded = s.get("loaded")
        if loaded:
            nbr, gender, outcome = loaded
            st.info(f"Test patient **{nbr}** ({gender}, age {s.age}). "
                    f"What really happened: **{'readmitted within 30 days' if outcome else 'not readmitted within 30 days'}**.")

        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            st.markdown("#### Model 1 · Clinical points score")
            st.metric("Chance of readmission", f"{risk1:.1%}",
                      f"{(risk1 - base) * 100:+.1f} points vs average ({base:.1%})", delta_color="inverse")
            (st.error if high1 else st.success)(
                f"{'🔴 HIGH RISK' if high1 else '🟢 LOW RISK'} · score {score1} of 19 "
                f"(high risk = {M['cutoff1']:.0f}+)")
            st.dataframe(pd.DataFrame({"Points": parts1}), use_container_width=True)

        with c2, st.container(border=True):
            st.markdown("#### Model 2 · Learned points score")
            st.metric("Chance of readmission", f"{risk2:.1%}",
                      f"{(risk2 - base) * 100:+.1f} points vs average ({base:.1%})", delta_color="inverse")
            getattr(st, tier2[2])(f"{tier2[1]} {tier2[0]} · score {score2} "
                                  f"(high risk = {M['cutoff2']:.0f}+)")
            st.dataframe(pd.DataFrame({"Condition": [pretty(c) for c, _ in fired] or ["(none true)"],
                                       "Points": [f"{p:+d}" for _, p in fired] or [""]}),
                         hide_index=True, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Why: the IF-THEN rule this patient matches")
            st.markdown("**IF** " + "  \n**AND** ".join(rule_text(steps, M["yes_no"])))
            st.markdown(f"**THEN** {leaf_risk:.1%} of {leaf_n:,} similar training patients were readmitted "
                        f"({leaf_risk / base:.1f}× the average).")

        with st.container(border=True):
            st.markdown("#### Suggested actions")
            st.caption("Hand-written suggestions, not learned by the model. For clinician review.")
            drivers = [c for c, p in fired if p > 0] + [f for f, _, above in steps if above]
            actions = [a for name, a in ACTIONS.items() if any(d.startswith(name) for d in drivers)]
            if tier2[0] == "LOW RISK" or not actions:
                st.write("No specific action triggered. Standard discharge process.")
            else:
                for a in actions:
                    st.checkbox(a, key=f"act_{a}")

# ---------------------------------------------------------------------------
# Tab 2: how the models work
# ---------------------------------------------------------------------------
with tab_how:
    st.subheader("Model 1: clinical points score (LACE-style)")
    st.write("A fixed score in the style of the LACE index used in hospitals. Add up the points; "
             f"a total of **{M['cutoff1']:.0f} or more** is flagged as high risk.")
    st.dataframe(MODEL1_TABLE, hide_index=True, use_container_width=True)

    st.subheader("Model 2: points score learned from the data")
    st.write("The data chose which conditions matter and how many points each is worth. "
             f"A total of **{M['cutoff2']:.0f} or more** is flagged as high risk.")
    col_a, col_b = st.columns(2)
    pts_table = pd.DataFrame({"Condition": [pretty(c) for c in M["points"].index], "Points": M["points"].values})
    col_a.dataframe(pts_table.sort_values("Points", ascending=False), hide_index=True, use_container_width=True)
    lo, hi = M["score2_range"]
    scores = np.arange(lo, hi + 1)
    col_b.dataframe(pd.DataFrame({"Score": scores,
                                  "Chance of readmission": [f"{p:.1%}" for p in
                                                            M["risk2"].predict_proba(scores.reshape(-1, 1))[:, 1]]}),
                    hide_index=True, use_container_width=True)

    st.subheader("Model 2: IF-THEN rules (highest risk first)")
    for n, (risk, size, steps) in enumerate(all_rules(M), start=1):
        with st.expander(f"Rule {n}: {risk:.1%} readmitted ({size:,} patients, {risk / M['base_rate']:.1f}× average)"):
            st.markdown("**IF** " + "  \n**AND** ".join(rule_text(steps, M["yes_no"])))

# ---------------------------------------------------------------------------
# Tab 3: model performance (results saved by the notebooks)
# ---------------------------------------------------------------------------
with tab_perf:
    st.write("Results on 19,773 test hospital stays the models never saw, saved by the notebooks.")
    comp = os.path.join(RESULTS, "comparison.csv")
    if os.path.exists(comp):
        st.dataframe(pd.read_csv(comp, index_col=0), use_container_width=True)
        st.caption("AUC-ROC: 0.50 = guessing, 1.00 = perfect. Accuracy alone is misleading, because 89% "
                   "of stays are not readmitted.")
    else:
        st.warning("comparison.csv not found. Run M2_learned_points_score.ipynb to create it.")
    for title, img in [("Model 1 charts", "model1_charts.png"), ("Model 2 charts", "model2_charts.png")]:
        path = os.path.join(RESULTS, img)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
