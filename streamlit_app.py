
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Diabetes Prediction – Model Bench", layout="wide")
sns.set_style("whitegrid")

# ------------- Sidebar Controls -------------
st.sidebar.title("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload CSV (expects 'Diabetes_binary' as target)", type=["csv"])
default_path = "diabetes dataset.csv"

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.20, 0.01)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 100, 1000, 300, 50)
gb_learning_rate = st.sidebar.select_slider("Gradient Boosting: learning_rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
threshold = st.sidebar.slider("Classification threshold", 0.1, 0.9, 0.5, 0.01)
topn_importance = st.sidebar.slider("Top-N feature importance", 5, 30, 15, 1)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Run this app with: `streamlit run streamlit_app.py`.\n\nIf no file is uploaded, the app looks for **diabetes dataset.csv** in the working directory.")

# ------------- Data Loading -------------
@st.cache_data(show_spinner=False)
def load_data(file_buffer, fallback_path):
    if file_buffer is not None:
        df = pd.read_csv(file_buffer)
    else:
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                "No file uploaded and default 'diabetes dataset.csv' not found in current directory."
            )
        df = pd.read_csv(fallback_path)
    return df

try:
    df = load_data(uploaded, default_path)
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

st.success(f"Loaded dataset with shape {df.shape}")
with st.expander("🔎 Preview & Columns", expanded=False):
    st.write(df.head())
    st.write("Columns:", list(df.columns))

if "Diabetes_binary" not in df.columns:
    st.error("Expected target column 'Diabetes_binary' not found!")
    st.stop()

# Split X, y and coerce numerics
y = df["Diabetes_binary"].astype(int)
X = df.drop(columns=["Diabetes_binary"]).copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Preprocess & models (constructed with current sidebar params)
numeric_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", numeric_preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
    ]),
    "Random Forest": Pipeline([
        ("prep", numeric_preprocess),
        ("clf", RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", numeric_preprocess),
        ("clf", GradientBoostingClassifier(random_state=random_state, learning_rate=gb_learning_rate))
    ])
}

# ------------- Training & Evaluation -------------
def evaluate_model(name, pipe, X_train, X_test, y_train, y_test, thr=0.5):
    pipe.fit(X_train, y_train)
    proba = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        scores = pipe.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    y_pred = (proba >= thr).astype(int) if proba is not None else pipe.predict(X_test)

    metrics = {
        "AUC": roc_auc_score(y_test, proba) if proba is not None else np.nan,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
        "proba": proba,
        "y_pred": y_pred
    }
    return metrics, pipe

# Cache training, ignoring unhashable 'models' dict
@st.cache_resource(show_spinner=True)
def train_all_models(_models_dict, X_train, X_test, y_train, y_test, thr):
    results = {}
    fitted = {}
    for name, pipe in _models_dict.items():
        res, fitted_pipe = evaluate_model(name, pipe, X_train, X_test, y_train, y_test, thr=thr)
        results[name] = res
        fitted[name] = fitted_pipe
    return results, fitted

with st.spinner("Training models..."):
    results, fitted = train_all_models(models, X_train, X_test, y_train, y_test, threshold)

# Metrics table
metrics_table = pd.DataFrame({
    mname: {
        "AUC": res["AUC"],
        "Accuracy": res["Accuracy"],
        "Precision": res["Precision"],
        "Recall": res["Recall"],
        "F1": res["F1"]
    } for mname, res in results.items()
}).T

# Helper to pick best by AUC
def pick_best(table):
    auc_vals = table["AUC"].fillna(-np.inf)
    best_by_auc = auc_vals.idxmax()
    return best_by_auc, table.loc[best_by_auc, "AUC"]

best_model_name, best_auc = pick_best(metrics_table)

# ------------- Layout: Tabs -------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 ROC & PR", "🧮 Confusion Matrices",
    "🌲 Feature Importance", "📋 Compare Metrics", "🔮 Predict Single Case"
])

with tab1:
    st.subheader("Overall Evaluation")
    st.dataframe(metrics_table.round(4))
    st.info(f"**Best model by AUC:** {best_model_name}  (AUC = {best_auc:.4f})")

with tab2:
    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, res in results.items():
            if res["proba"] is not None:
                fpr, tpr, _ = roc_curve(y_test, res["proba"])
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random (AUC = 0.500)")
        ax.set_title("ROC Curves")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        st.pyplot(fig)

    with col_pr:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for name, res in results.items():
            if res["proba"] is not None:
                prec, rec, _ = precision_recall_curve(y_test, res["proba"])
                ax2.plot(rec, prec, lw=2, label=f"{name}")
        baseline = y_test.mean()
        ax2.hlines(baseline, 0, 1, linestyles="--", label=f"Prevalence = {baseline:.2f}")
        ax2.set_title("Precision–Recall Curves")
        ax2.set_xlabel("Recall (Sensitivity)")
        ax2.set_ylabel("Precision (PPV)")
        ax2.legend(loc="best", frameon=True, fontsize=8)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        st.pyplot(fig2)

with tab3:
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        cm = res["ConfusionMatrix"]
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    annot_kws={"size": 12})
        ax_cm.set_title(f"{name}")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_xticklabels(["0 (No Diabetes)", "1 (Diabetes)"], rotation=20)
        ax_cm.set_yticklabels(["0 (No Diabetes)", "1 (Diabetes)"], rotation=0)
        cols[idx].pyplot(fig_cm)

with tab4:
    st.subheader("Tree-Based Feature Importances")
    def plot_feature_importance(pipe, title, feature_names, top_n=15):
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=feature_names)
            top = importances.sort_values(ascending=False).head(top_n)[::-1]
            fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
            ax_imp.barh(top.index, top.values)
            ax_imp.set_title(f"Top {top_n} Features: {title}")
            ax_imp.set_xlabel("Importance (Gini decrease)")
            ax_imp.set_ylabel("Feature")
            fig_imp.tight_layout()
            return fig_imp
        return None

    for name in ["Random Forest", "Gradient Boosting"]:
        fig_imp = plot_feature_importance(fitted[name], name, X.columns, top_n=topn_importance)
        if fig_imp is not None:
            st.pyplot(fig_imp)
        else:
            st.warning(f"{name} has no attribute 'feature_importances_'")

with tab5:
    st.subheader("Metric Comparison")
    comp = metrics_table[["AUC", "Accuracy", "F1"]].copy()
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
    comp.plot(kind="bar", rot=0, ax=ax_bar)
    ax_bar.set_title("Model Comparison: AUC, Accuracy, F1")
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("Score")
    ax_bar.set_ylim(0, 1)
    ax_bar.legend(title="Metric", loc="lower right", frameon=True, fontsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)

with tab6:
    st.subheader("Predict for a Single Person")
    st.caption("Binary features (0/1) are shown as dropdowns with labels. "
               "Other features are text boxes with **observed ranges** from the training data. "
               "Blank/invalid inputs fall back to the training **median**.")

    # Semantic labels for common binary fields (applied only if the column exists)
    BINARY_LABELS = {
        "Sex": {0: "Female (0)", 1: "Male (1)"},
        "HighBP": {0: "No high blood pressure (0)", 1: "High blood pressure (1)"},
        "HighChol": {0: "No high cholesterol (0)", 1: "High cholesterol (1)"},
        "CholCheck": {0: "No cholesterol check (0)", 1: "Had cholesterol check (1)"},
        "Smoker": {0: "Non-smoker (0)", 1: "Smoker (1)"},
        "Stroke": {0: "No stroke history (0)", 1: "Stroke history (1)"},
        "HeartDiseaseorAttack": {0: "No heart disease/attack (0)", 1: "Heart disease/attack (1)"},
        "PhysActivity": {0: "No physical activity (0)", 1: "Physically active (1)"},
        "Fruits": {0: "<1 serving/day (0)", 1: "≥1 serving/day (1)"},
        "Veggies": {0: "<1 serving/day (0)", 1: "≥1 serving/day (1)"},
        "HvyAlcoholConsump": {0: "No (0)", 1: "Heavy alcohol consumption (1)"},
        "AnyHealthcare": {0: "No healthcare access (0)", 1: "Has healthcare access (1)"},
        "NoDocbcCost": {0: "No cost barrier (0)", 1: "Could not see doctor due to cost (1)"},
        "DiffWalk": {0: "No difficulty walking (0)", 1: "Difficulty walking (1)"},
        # Add more if your dataset has other named binaries
    }

    defaults = X_train.median(numeric_only=True)
    mins = X_train.min(numeric_only=True)
    maxs = X_train.max(numeric_only=True)
    p5 = X_train.quantile(0.05, numeric_only=True)
    p95 = X_train.quantile(0.95, numeric_only=True)

    # Detect binary columns (unique values subset of {0,1})
    binary_cols = set()
    for col in X.columns:
        vals = pd.Series(X[col]).dropna().unique()
        if len(vals) > 0:
            try:
                vals = np.array(vals, dtype=float)
                unique_set = set(np.unique(vals).tolist())
                if unique_set.issubset({0.0, 1.0}):
                    binary_cols.add(col)
            except Exception:
                pass

    cols_layout = st.columns(3)
    user_input_raw = {}

    for i, col in enumerate(X.columns):
        with cols_layout[i % 3]:
            if col in binary_cols:
                # Default selection from training median
                dflt = defaults.get(col, 0.0)
                try:
                    dflt = int(round(float(dflt)))
                except Exception:
                    dflt = 0
                dflt = 1 if dflt == 1 else 0

                # Build a format function using semantic labels if available
                label_map = BINARY_LABELS.get(col, {0: "No (0)", 1: "Yes (1)"})
                fmt = lambda v, m=label_map: m.get(v, str(v))

                choice = st.selectbox(
                    col,
                    options=[0, 1],
                    index=[0, 1].index(dflt),
                    format_func=fmt,
                    help=label_map.get(1, "1 = Yes") + " | " + label_map.get(0, "0 = No")
                )
                user_input_raw[col] = choice
            else:
                # Build help text with observed ranges
                mn = float(mins.get(col, np.nan)) if col in mins.index else np.nan
                mx = float(maxs.get(col, np.nan)) if col in maxs.index else np.nan
                q5 = float(p5.get(col, np.nan)) if col in p5.index else np.nan
                q95 = float(p95.get(col, np.nan)) if col in p95.index else np.nan
                med = float(defaults.get(col, 0.0)) if col in defaults.index else 0.0

                rng_parts = []
                if not np.isnan(mn) and not np.isnan(mx):
                    rng_parts.append(f"Observed range: **{mn:.3g}–{mx:.3g}**")
                if not np.isnan(q5) and not np.isnan(q95):
                    rng_parts.append(f"Typical (5–95%): **{q5:.3g}–{q95:.3g}**")
                rng_parts.append(f"Median: **{med:.3g}**")
                help_str = " | ".join(rng_parts)

                placeholder = "" if pd.isna(med) else str(med)
                txt = st.text_input(col, value="", placeholder=placeholder, help=help_str)
                user_input_raw[col] = txt

    # Parse into model-ready row
    parsed = {}
    parse_warnings = []
    for col in X.columns:
        val = user_input_raw[col]
        if col in binary_cols:
            parsed[col] = int(val)
        else:
            if val is None or str(val).strip() == "":
                parsed[col] = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
            else:
                try:
                    parsed[col] = float(val)
                except Exception:
                    parsed[col] = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
                    parse_warnings.append(col)

    input_df = pd.DataFrame([parsed])
    chosen_model = st.selectbox("Choose model for inference", list(models.keys()), index=list(models.keys()).index(best_model_name))

    if st.button("Predict"):
        if parse_warnings:
            st.warning(f"These fields were non-numeric and were replaced by their median: {', '.join(parse_warnings)}")

        mdl = fitted[chosen_model]
        if hasattr(mdl.named_steps["clf"], "predict_proba"):
            proba = mdl.predict_proba(input_df)[0, 1]
            pred = int(proba >= threshold)
        elif hasattr(mdl.named_steps["clf"], "decision_function"):
            score = mdl.decision_function(input_df)[0]
            proba = 1 / (1 + np.exp(-score))
            pred = int(proba >= threshold)
        else:
            pred = int(mdl.predict(input_df)[0])
            proba = float(pred)

        if pred == 1:
            st.error("Prediction: **Diabetes**")
        else:
            st.success("Prediction: **No Diabetes**")
