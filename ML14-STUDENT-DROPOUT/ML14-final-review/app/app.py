"""Student dropout early warning system with fairness constraints (Model 1 and Model 2).

Run from this folder with:   streamlit run app.py
The models are trained once from data.csv, exactly as in
Student_dropout.ipynb (Model 1) and Student_dropout2.ipynb (Model 2).
"""
import os
import random
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "ML14-second-review", "data", "data.csv")
RANDOM_STATE = 42
GAP_LIMIT = 0.05
TARGET_RECALL_1 = 0.75     # Model 1 (at enrollment)
TARGET_RECALL_2 = 0.85     # Model 2 (after first semester)
# Best settings found by RandomizedSearchCV in Student_dropout2.ipynb
HGB_PARAMS = dict(min_samples_leaf=20, max_leaf_nodes=31, max_iter=300, max_features=0.5,
                  learning_rate=0.02, l2_regularization=0.0)

# ---------------------------------------------------------------------------
# Shared definitions (copied from the notebooks)
# ---------------------------------------------------------------------------
SEM1 = "Curricular units 1st sem"
ENROLLMENT_FEATURES = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Scholarship holder", "Age at enrollment",
    "International", "Unemployment rate", "Inflation rate", "GDP",
]
SEM1_FEATURES = [f"{SEM1} (credited)", f"{SEM1} (enrolled)", f"{SEM1} (evaluations)",
                 f"{SEM1} (approved)", f"{SEM1} (grade)", f"{SEM1} (without evaluations)"]
CATEGORICAL = ["Marital status", "Application mode", "Course", "Previous qualification",
               "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
NUMERIC_1 = [f for f in ENROLLMENT_FEATURES if f not in CATEGORICAL]
ENGINEERED = ["sem1_approval_rate", "sem1_pass_per_exam", "sem1_no_exam_share",
              "sem1_failed_units", "fee_problem", "grade_gap"]
NUMERIC_2 = NUMERIC_1 + SEM1_FEATURES + ENGINEERED
FEATURES_2 = CATEGORICAL + NUMERIC_2
RAW_INPUTS = ENROLLMENT_FEATURES + SEM1_FEATURES

COURSES = {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
           8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
           9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
           9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
           9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
           9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)"}
MARITAL = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"}
APPLICATION = {1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
               5: "1st phase - special contingent (Azores)", 7: "Holders of other higher courses",
               10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
               16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
               18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99, item b2",
               27: "Ordinance No. 533-A/99, item b3", 39: "Over 23 years old", 42: "Transfer",
               43: "Change of course", 44: "Technological specialization diploma holders",
               51: "Change of institution/course", 53: "Short cycle diploma holders",
               57: "Change of institution/course (International)"}

# Plain names for the explanation, and suggested actions written by hand (NOT learned by the model).
# Have student-services staff review these before using them anywhere real.
FACTOR_NAMES = {
    f"{SEM1} (approved)": "Few first-semester units passed",
    f"{SEM1} (grade)": "Low first-semester grades",
    f"{SEM1} (enrolled)": "Light first-semester course load",
    f"{SEM1} (evaluations)": "Few first-semester exams attempted",
    f"{SEM1} (without evaluations)": "Units without any exam",
    f"{SEM1} (credited)": "Credited units",
    "Tuition fees up to date": "Tuition fees not up to date",
    "Debtor": "Owes money to the institution",
    "Scholarship holder": "No scholarship",
    "Course": "Course with a high dropout rate",
    "Application mode": "Entry route",
    "Age at enrollment": "Age at enrollment",
    "Admission grade": "Admission grade",
    "Previous qualification (grade)": "Previous qualification grade",
    "Marital status": "Marital status",
    "Daytime/evening attendance": "Evening attendance",
    "Displaced": "Living away from home",
}
ACTIONS = {
    f"{SEM1} (approved)": "Offer tutoring and an academic-plan meeting.",
    f"{SEM1} (grade)": "Offer tutoring or study-skills support.",
    f"{SEM1} (enrolled)": "Check whether the student is disengaging or working alongside study.",
    f"{SEM1} (evaluations)": "Early check-in by the tutor about attendance and exams.",
    f"{SEM1} (without evaluations)": "Contact the student about missed exams.",
    "Tuition fees up to date": "Refer to the financial-aid office for a payment plan or emergency grant.",
    "Debtor": "Refer to financial aid and check for hardship funds.",
    "Scholarship holder": "Check eligibility for scholarships or bursaries.",
    "Course": "Connect the student with a course mentor or peer group.",
    "Application mode": "Offer the bridging / returning-learner programme.",
    "Admission grade": "Offer a first-year transition programme.",
    "Age at enrollment": "Offer flexible timetabling and mature-student support.",
    "Daytime/evening attendance": "Check workload and offer evening-student support.",
}


def load_data(path=DATA):
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:              # Kaggle version uses commas
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["dropout"] = (df["Target"] == "Dropout").astype(int)
    df["Gender name"] = df["Gender"].map({0: "Female", 1: "Male"})
    df["Age band"] = age_band(df["Age at enrollment"])
    return df


def age_band(age):
    return pd.cut(pd.Series(age), [0, 20, 24, 200], labels=["20 or under", "21-24", "25 and over"]).astype(str).values


def add_features(df):
    """Model 2's extra features, computed from the first-semester results."""
    df = df.copy()
    enrolled = df[f"{SEM1} (enrolled)"]
    df["sem1_approval_rate"] = np.where(enrolled > 0, df[f"{SEM1} (approved)"] / enrolled.clip(lower=1), 0)
    df["sem1_pass_per_exam"] = np.where(df[f"{SEM1} (evaluations)"] > 0,
                                        df[f"{SEM1} (approved)"] / df[f"{SEM1} (evaluations)"].clip(lower=1), 0)
    df["sem1_no_exam_share"] = np.where(enrolled > 0, df[f"{SEM1} (without evaluations)"] / enrolled.clip(lower=1), 0)
    df["sem1_failed_units"] = (enrolled - df[f"{SEM1} (approved)"]).clip(lower=0)
    df["fee_problem"] = ((df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)).astype(int)
    df["grade_gap"] = df["Admission grade"] - df["Previous qualification (grade)"]
    return df


def reweighing_weights(y, group):
    w = np.zeros(len(y))
    for g in np.unique(group):
        for label in [0, 1]:
            mask = (group == g) & (y == label)
            w[mask] = (group == g).mean() * (y == label).mean() / mask.mean()
    return w


def threshold_for_recall(y_true, prob, target):
    """Highest cut-off that still catches at least `target` of the dropouts."""
    for t in np.arange(0.95, 0.04, -0.01):
        if recall_score(y_true, (prob >= t).astype(int)) >= target:
            return round(float(t), 2)
    return 0.05


def recall_gap(y_true, y_pred, group):
    r = [y_pred[(group == g) & (y_true == 1)].mean() for g in np.unique(group)]
    return max(r) - min(r)


def group_report(y_true, y_pred, group):
    rows = []
    for g in sorted(np.unique(group), key=str):
        m = group == g
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        rows.append({"Group": g, "Students": int(m.sum()), "Dropouts": int(tp + fn),
                     "Flagged": y_pred[m].mean(), "Dropouts caught (recall)": tp / (tp + fn),
                     "False alarm rate": fp / (fp + tn), "Precision": tp / (tp + fp) if (tp + fp) else 0})
    return pd.DataFrame(rows).set_index("Group")


def all_metrics(y_true, pred, prob):
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred),
            "Recall (dropouts caught)": recall_score(y_true, pred), "F1 score": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, prob), "Students flagged": pred.mean()}


# ---------------------------------------------------------------------------
# Training (once)
# ---------------------------------------------------------------------------
def model1_pipeline():
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC_1),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000))])


@st.cache_resource(show_spinner=False)
def train():
    df = add_features(load_data())
    y = df["dropout"]
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    tr, te = df.loc[idx_train], df.loc[idx_test]
    y_tr, y_te = y.loc[idx_train].values, y.loc[idx_test].values
    group_tr = (tr["Gender name"] + ", " + tr["Age band"]).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    M = {"df": df, "test_idx": list(idx_test), "y_test": y_te,
         "gender_test": te["Gender name"].values, "age_test": te["Age band"].values,
         "categories": {c: sorted(int(v) for v in df[c].unique()) for c in CATEGORICAL},
         "typical": {c: (tr[c].mode()[0] if c in CATEGORICAL or tr[c].nunique() <= 2 else tr[c].median())
                     for c in RAW_INPUTS}}

    # Model 1: logistic regression at enrollment, reweighed by gender x age band
    w = reweighing_weights(y_tr, group_tr)
    m1 = model1_pipeline()
    oof1 = cross_val_predict(m1, tr[ENROLLMENT_FEATURES], y_tr, cv=cv, method="predict_proba",
                             params={"model__sample_weight": w})[:, 1]
    M["t1"] = threshold_for_recall(y_tr, oof1, TARGET_RECALL_1)
    M["m1"] = m1.fit(tr[ENROLLMENT_FEATURES], y_tr, model__sample_weight=w)

    # Model 2: gradient boosting after the first semester, with equal-opportunity thresholds
    m2 = HistGradientBoostingClassifier(categorical_features="from_dtype", class_weight="balanced",
                                        early_stopping=False, random_state=RANDOM_STATE, **HGB_PARAMS)
    X_tr = to_model2(tr, M["categories"])
    oof2 = cross_val_predict(m2, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    M["t2_single"] = threshold_for_recall(y_tr, oof2, TARGET_RECALL_2)
    single = (oof2 >= M["t2_single"]).astype(int)
    attrs = {"Gender": tr["Gender name"].values, "Age band": tr["Age band"].values}
    M["oof_gaps"] = {k: recall_gap(y_tr, single, v) for k, v in attrs.items()}
    M["constrained"] = [k for k, gap in M["oof_gaps"].items() if gap > GAP_LIMIT]
    g2 = constraint_group(tr["Gender name"].values, tr["Age band"].values, M["constrained"])
    M["t2"] = {g: threshold_for_recall(y_tr[g2 == g], oof2[g2 == g], TARGET_RECALL_2) for g in np.unique(g2)}
    M["m2"] = m2.fit(X_tr, y_tr)

    # Test-set predictions (students never used for training)
    M["p1_test"] = M["m1"].predict_proba(te[ENROLLMENT_FEATURES])[:, 1]
    M["p2_test"] = M["m2"].predict_proba(to_model2(te, M["categories"]))[:, 1]
    g2_te = constraint_group(te["Gender name"].values, te["Age band"].values, M["constrained"])
    M["t2_test"] = pd.Series(g2_te).map(M["t2"]).values
    return M


def to_model2(df, categories):
    X = add_features(df)[FEATURES_2].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def constraint_group(gender, age, constrained):
    if not constrained:
        return np.full(len(gender), "All")
    parts = {"Gender": np.asarray(gender, dtype=object), "Age band": np.asarray(age, dtype=object)}
    out = parts[constrained[0]]
    for name in constrained[1:]:
        out = out + ", " + parts[name]
    return out.astype(str)


def predict(M, student):
    """student: one-row DataFrame with RAW_INPUTS and 'Gender'."""
    p1 = M["m1"].predict_proba(student[ENROLLMENT_FEATURES])[0, 1]
    p2 = M["m2"].predict_proba(to_model2(student, M["categories"]))[0, 1]
    gender = "Male" if int(student["Gender"].iloc[0]) == 1 else "Female"
    band = age_band(student["Age at enrollment"])[0]
    group = constraint_group([gender], [band], M["constrained"])[0]
    return p1, p2, M["t2"][group], group


def risk_factors(M, student, which, top=3):
    """What raises this student's risk: put each input back to a typical value and see how much risk falls."""
    cols = ENROLLMENT_FEATURES if which == 1 else RAW_INPUTS
    base = predict(M, student)[0 if which == 1 else 1]
    out = {}
    for c in cols:
        if c not in FACTOR_NAMES or student[c].iloc[0] == M["typical"][c]:
            continue
        s = student.copy()
        s[c] = M["typical"][c]
        out[c] = base - predict(M, s)[0 if which == 1 else 1]
    s = pd.Series(out)
    return s[s >= 0.02].sort_values(ascending=False).head(top)


def risk_level(p, t):
    if p >= max(0.7, t + 0.2):
        return "High"
    if p >= t:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dropout Early Warning", page_icon="🎓", layout="wide")
st.title("🎓 Student dropout early warning")
st.caption("UCI 'Predict Students' Dropout and Academic Success' dataset (4,424 students). "
           "A study project: not for real decisions about students.")

if not os.path.exists(DATA):
    st.error(f"Could not find data.csv in ML14-second-review/data/ ({DATA}).")
    st.stop()

with st.spinner("Training the models on data.csv (only the first time, about 30 seconds)..."):
    M = train()

GENDERS = {"Female": 0, "Male": 1}
YES_NO = {"No": 0, "Yes": 1}
FORM_KEYS = {  # session-state key -> data column
    "gender": "Gender", "age": "Age at enrollment", "course": "Course", "marital": "Marital status",
    "app_mode": "Application mode", "app_order": "Application order", "evening": "Daytime/evening attendance",
    "prev_qual": "Previous qualification", "prev_grade": "Previous qualification (grade)",
    "adm_grade": "Admission grade", "displaced": "Displaced", "special": "Educational special needs",
    "debtor": "Debtor", "fees": "Tuition fees up to date", "scholar": "Scholarship holder",
    "intl": "International", "mq": "Mother's qualification", "fq": "Father's qualification",
    "mo": "Mother's occupation", "fo": "Father's occupation", "unemp": "Unemployment rate",
    "infl": "Inflation rate", "gdp": "GDP", "s1_cred": f"{SEM1} (credited)", "s1_enr": f"{SEM1} (enrolled)",
    "s1_eval": f"{SEM1} (evaluations)", "s1_appr": f"{SEM1} (approved)", "s1_grade": f"{SEM1} (grade)",
    "s1_noeval": f"{SEM1} (without evaluations)",
}


BOOL_KEYS = {"displaced", "special", "debtor", "fees", "scholar", "intl"}
FLOAT_KEYS = {"prev_grade", "adm_grade", "s1_grade", "unemp", "infl", "gdp"}


def widget_value(key, v):
    """Widgets need plain Python types: bool for toggles, float or int for number boxes."""
    if key in BOOL_KEYS:
        return bool(v)
    if key in FLOAT_KEYS:
        return float(v)
    return int(v)


def defaults():
    d = {k: M["typical"][col] for k, col in FORM_KEYS.items() if col in M["typical"]}
    d.update(gender=0, age=20, s1_enr=6, s1_eval=7, s1_appr=5, s1_grade=12.5, s1_cred=0, s1_noeval=0)
    return {k: widget_value(k, v) for k, v in d.items()}


for key, value in defaults().items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("loaded", None)


def clear_loaded():
    st.session_state["loaded"] = None


def reset_form():
    for key, value in defaults().items():
        st.session_state[key] = value
    clear_loaded()


def load_test_student():
    """Fill the form with a random student from the test set (never used for training)."""
    i = random.choice(M["test_idx"])
    row = M["df"].loc[i]
    for key, col in FORM_KEYS.items():
        st.session_state[key] = widget_value(key, row[col])
    st.session_state["loaded"] = (int(i), row["Target"])


def code_select(label, key, names, options):
    st.selectbox(label, options, key=key, format_func=lambda c: names.get(c, f"Code {c}"), on_change=clear_loaded)


tab_check, tab_list, tab_fair, tab_perf = st.tabs(["Check a student", "Watch list", "Fairness", "Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: check a student
# ---------------------------------------------------------------------------
with tab_check:
    left, right = st.columns([1, 2], gap="large")

    with left:
        b1, b2 = st.columns(2)
        b1.button("🎲 Load a test student", on_click=load_test_student, use_container_width=True)
        b2.button("↺ Reset", on_click=reset_form, use_container_width=True)
        kw = {"on_change": clear_loaded}

        st.subheader("Student details")
        st.radio("Gender", [0, 1], key="gender", format_func=lambda v: "Female" if v == 0 else "Male",
                 horizontal=True, **kw)
        st.number_input("Age at enrollment", 17, 70, key="age", **kw)
        code_select("Course", "course", COURSES, M["categories"]["Course"])

        with st.expander("Enrollment", expanded=True):
            code_select("Entry route (application mode)", "app_mode", APPLICATION, M["categories"]["Application mode"])
            code_select("Marital status", "marital", MARITAL, M["categories"]["Marital status"])
            st.number_input("Admission grade (0-200)", 95.0, 190.0, key="adm_grade", step=1.0, **kw)
            st.number_input("Previous qualification grade (0-200)", 95.0, 190.0, key="prev_grade", step=1.0, **kw)
            st.radio("Attendance", [1, 0], key="evening", format_func=lambda v: "Daytime" if v == 1 else "Evening",
                     horizontal=True, **kw)

        with st.expander("Money and circumstances", expanded=True):
            st.toggle("Tuition fees up to date", key="fees", **kw)
            st.toggle("Owes money to the institution (debtor)", key="debtor", **kw)
            st.toggle("Scholarship holder", key="scholar", **kw)
            st.toggle("Living away from home (displaced)", key="displaced", **kw)

        with st.expander("First semester results (used by Model 2 only)", expanded=True):
            st.number_input("Units enrolled", 0, 26, key="s1_enr", **kw)
            st.number_input("Exams attempted", 0, 45, key="s1_eval", **kw)
            st.number_input("Units passed", 0, 26, key="s1_appr", **kw)
            st.number_input("Average grade (0-20)", 0.0, 20.0, key="s1_grade", step=0.5, **kw)
            st.number_input("Units without any exam", 0, 12, key="s1_noeval", **kw)
            st.number_input("Units credited from before", 0, 20, key="s1_cred", **kw)

        with st.expander("Other details (codes from the dataset)"):
            st.number_input("Application order (1 = first choice)", 0, 9, key="app_order", **kw)
            code_select("Previous qualification", "prev_qual", {}, M["categories"]["Previous qualification"])
            code_select("Mother's qualification", "mq", {}, M["categories"]["Mother's qualification"])
            code_select("Father's qualification", "fq", {}, M["categories"]["Father's qualification"])
            code_select("Mother's occupation", "mo", {}, M["categories"]["Mother's occupation"])
            code_select("Father's occupation", "fo", {}, M["categories"]["Father's occupation"])
            st.toggle("Educational special needs", key="special", **kw)
            st.toggle("International student", key="intl", **kw)
            st.number_input("Unemployment rate (%)", 0.0, 20.0, key="unemp", step=0.1, **kw)
            st.number_input("Inflation rate (%)", -5.0, 10.0, key="infl", step=0.1, **kw)
            st.number_input("GDP growth", -10.0, 10.0, key="gdp", step=0.01, **kw)

    s = st.session_state
    student = pd.DataFrame([{col: int(s[k]) if isinstance(s[k], bool) else s[k] for k, col in FORM_KEYS.items()}])
    problems = []
    if s["s1_appr"] > s["s1_enr"]:
        problems.append("Units passed is more than units enrolled.")
    p1, p2, t2, group2 = predict(M, student)

    with right:
        if s["loaded"]:
            i, outcome = s["loaded"]
            st.info(f"Test student #{i}. Actual outcome (for checking only): **{outcome}**")
        for msg in problems:
            st.warning(msg)

        c1, c2 = st.columns(2, gap="medium")
        for col, title, sub, p, t, note in [
            (c1, "Model 1", "At enrollment (logistic regression, reweighed for fairness)", p1, M["t1"],
             "one cut-off for everyone"),
            (c2, "Model 2", "After first semester (gradient boosting, fair cut-offs)", p2, t2,
             f"cut-off for {group2}" if group2 != "All" else "one cut-off for everyone")]:
            with col:
                st.subheader(title)
                st.caption(sub)
                level = risk_level(p, t)
                st.metric("Chance of dropping out", f"{p:.0%}", f"{level} risk",
                          delta_color={"High": "inverse", "Medium": "off", "Low": "normal"}[level])
                st.progress(min(float(p), 1.0))
                if p >= t:
                    st.error(f"FLAG for advisor contact (cut-off {t:.0%}, {note})")
                else:
                    st.success(f"No flag (cut-off {t:.0%}, {note})")

        st.subheader("Why, and what to do")
        which = 2 if p2 >= t2 or p2 >= p1 else 1
        factors = risk_factors(M, student, which)
        st.caption(f"Based on Model {which}. Each factor shows how much the risk would fall if that detail "
                   "were typical for students in the data.")
        if factors.empty:
            st.write("Nothing in this student's details raises the risk noticeably.")
        for c, drop in factors.items():
            st.markdown(f"**{FACTOR_NAMES[c]}**: +{drop:.0%} risk")
            if c in ACTIONS:
                st.markdown(f"&nbsp;&nbsp;→ {ACTIONS[c]}")
        st.caption("Suggested actions are written by hand, not learned by the model.")

# ---------------------------------------------------------------------------
# Tab 2: watch list (test students)
# ---------------------------------------------------------------------------
with tab_list:
    st.write("Students from the test set (never used for training), highest risk first.")
    which = st.radio("Model", ["Model 2 (after first semester)", "Model 1 (at enrollment)"], horizontal=True)
    df_te = M["df"].loc[M["test_idx"]]
    if which.startswith("Model 2"):
        p, t = M["p2_test"], M["t2_test"]
    else:
        p, t = M["p1_test"], np.full(len(df_te), M["t1"])
    view = pd.DataFrame({"Student #": df_te.index, "Course": df_te["Course"].map(COURSES),
                         "Gender": df_te["Gender name"].values, "Age band": df_te["Age band"].values,
                         "Dropout risk": p, "Cut-off": t,
                         "Risk level": [risk_level(a, b) for a, b in zip(p, t)],
                         "Actual outcome": df_te["Target"].values})
    f1, f2, f3 = st.columns(3)
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    course = f2.selectbox("Course", ["All"] + sorted(view["Course"].dropna().unique()))
    band = f3.selectbox("Age band", ["All", "20 or under", "21-24", "25 and over"])
    view = view[view["Risk level"].isin(levels)]
    if course != "All":
        view = view[view["Course"] == course]
    if band != "All":
        view = view[view["Age band"] == band]
    st.write(f"{len(view)} students")
    st.dataframe(view.sort_values("Dropout risk", ascending=False)
                 .style.format({"Dropout risk": "{:.0%}", "Cut-off": "{:.0%}"}),
                 hide_index=True, use_container_width=True, height=500)

# ---------------------------------------------------------------------------
# Tab 3: fairness
# ---------------------------------------------------------------------------
with tab_fair:
    st.write("**Fairness constraint (equal opportunity):** among students who will drop out, every group "
             f"should be caught at the same rate (recall gap ≤ {GAP_LIMIT}). Flag rates may differ, because "
             "real dropout rates differ between groups.")
    y_te = M["y_test"]
    attr = st.radio("Compare groups by", ["Age band", "Gender"], horizontal=True)
    groups = M["age_test"] if attr == "Age band" else M["gender_test"]

    options = {
        "Model 1 (reweighed, at enrollment)": (M["p1_test"] >= M["t1"]).astype(int),
        f"Model 2, one cut-off ({M['t2_single']:.2f})": (M["p2_test"] >= M["t2_single"]).astype(int),
        "Model 2, fair cut-offs": (M["p2_test"] >= M["t2_test"]).astype(int),
    }
    summary = pd.DataFrame({name: {"Recall gap": recall_gap(y_te, pred, groups),
                                   "Lowest group recall": group_report(y_te, pred, groups)["Dropouts caught (recall)"].min(),
                                   "Overall recall": recall_score(y_te, pred),
                                   "Precision": precision_score(y_te, pred),
                                   "Students flagged": pred.mean()}
                            for name, pred in options.items()}).T
    st.dataframe(summary.style.format("{:.3f}").apply(
        lambda col: ["color: #c0392b" if (col.name == "Recall gap" and v > GAP_LIMIT) else "" for v in col]),
        use_container_width=True)

    st.subheader(f"Share of dropouts caught, by {attr.lower()}")
    st.bar_chart(pd.DataFrame({name: group_report(y_te, pred, groups)["Dropouts caught (recall)"]
                               for name, pred in options.items()}), stack=False)
    pick = st.selectbox("Details for", list(options))
    st.dataframe(group_report(y_te, options[pick], groups).style.format(
        {c: "{:.1%}" for c in ["Flagged", "Dropouts caught (recall)", "False alarm rate", "Precision"]}),
        use_container_width=True)

    st.subheader("Model 2's cut-offs")
    gaps = ", ".join(f"{k} {v:.3f}" for k, v in M["oof_gaps"].items())
    st.write(f"With one cut-off, the recall gap on the training students was: {gaps}. "
             f"Groups above {GAP_LIMIT} get their own cut-off: "
             f"**{' x '.join(M['constrained']) if M['constrained'] else 'none needed'}**.")
    st.dataframe(pd.Series(M["t2"], name="Flag if risk ≥").sort_index(), use_container_width=True)
    st.caption("The test set has only ~100-140 dropouts per group, so differences of a few points can be noise. "
               "Gender is never one of Model 2's risk inputs; it and the age band are used only to choose the "
               "cut-off. Check with your institution before using group-specific cut-offs.")

# ---------------------------------------------------------------------------
# Tab 4: model performance
# ---------------------------------------------------------------------------
with tab_perf:
    y_te = M["y_test"]
    pred1 = (M["p1_test"] >= M["t1"]).astype(int)
    pred2 = (M["p2_test"] >= M["t2_test"]).astype(int)
    perf = pd.DataFrame({"Model 1 (at enrollment)": all_metrics(y_te, pred1, M["p1_test"]),
                         "Model 2 (after first semester)": all_metrics(y_te, pred2, M["p2_test"])})
    st.write(f"Test set: {len(y_te)} students never used for training ({y_te.sum()} dropouts).")
    st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)

    c1, c2 = st.columns(2)
    for col, name, pred in [(c1, "Model 1", pred1), (c2, "Model 2", pred2)]:
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        col.subheader(name)
        col.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: not dropout", "Actual: dropout"],
                                   columns=["Not flagged", "Flagged"]), use_container_width=True)
    st.caption(f"Model 1 aims to catch {TARGET_RECALL_1:.0%} of dropouts on enrollment-day information only; "
               f"Model 2 aims for {TARGET_RECALL_2:.0%} once first-semester results are known. "
               "Cut-offs are chosen on the training students, never on the test set.")
