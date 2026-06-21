import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stMetric {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Data + model loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_scaled"] = StandardScaler().fit_transform(df[["Amount"]])
    return df


@st.cache_resource
def train_models(df):
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Hour"]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return X_test, y_test, log_reg, rf, feature_cols


df = load_data()

with st.spinner("Training models..."):
    X_test, y_test, log_reg, rf, feature_cols = train_models(df)

models = {"Logistic Regression": log_reg, "Random Forest": rf}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Kaggle ULB Credit Card Fraud dataset")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox("Model", list(models.keys()))
threshold = st.sidebar.slider("Classification threshold", 0.0, 1.0, 0.50, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset summary**")
st.sidebar.write(f"Transactions: `{len(df):,}`")
st.sidebar.write(f"Fraud cases: `{int(df['Class'].sum()):,}`")
st.sidebar.write(f"Fraud rate: `{df['Class'].mean()*100:.4f}%`")
st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn + Streamlit")

model = models[model_choice]
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# ----------------------------------------------------------------------------
# Header + KPI row
# ----------------------------------------------------------------------------
st.title("Credit Card Fraud Detection")
st.caption(f"Model: **{model_choice}**  ·  Threshold: **{threshold:.2f}**")

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("ROC-AUC", f"{roc_auc:.3f}")
k2.metric("PR-AUC", f"{pr_auc:.3f}")
k3.metric("Precision", f"{prec:.3f}")
k4.metric("Recall", f"{rec:.3f}")
k5.metric("F1 score", f"{f1:.3f}")
k6.metric("Fraud caught", f"{tp} / {tp+fn}")

st.markdown("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Performance", "Feature Insights", "Data Explorer"])

# ---- TAB 1: Overview ----
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Class Distribution")
        counts = df["Class"].value_counts().rename({0: "Legit", 1: "Fraud"})
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color=counts.index, color_discrete_map={"Legit": "#2ecc71", "Fraud": "#e74c3c"})
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Transactions by Hour")
        hour_df = df.groupby(["Hour", "Class"]).size().reset_index(name="count")
        hour_df["Class"] = hour_df["Class"].map({0: "Legit", 1: "Fraud"})
        fig = px.line(hour_df, x="Hour", y="count", color="Class",
                      color_discrete_map={"Legit": "#2ecc71", "Fraud": "#e74c3c"})
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.subheader("Amount Distribution (legit vs fraud)")
        fig = px.box(df, x="Class", y="Amount", color="Class",
                     color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                     points=False)
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Legit", "Fraud"])
        fig.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Top Correlations with Fraud")
        corr = df.drop(columns=["Hour", "Amount_scaled"]).corr()["Class"].drop("Class")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
        fig = px.bar(corr[::-1], orientation="h",
                     color=corr[::-1].values, color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: Model performance ----
with tab2:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Confusion Matrix")
        fig = px.imshow(cm, text_auto=",d", color_continuous_scale="Blues",
                         x=["Pred: Legit", "Pred: Fraud"], y=["True: Legit", "True: Fraud"])
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(width=3, color="#3498db"),
                                  name=model_choice))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(height=380, margin=dict(t=10),
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader("Precision-Recall Curve")
        p, r, _ = precision_recall_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", line=dict(width=3, color="#e74c3c")))
        fig.add_hline(y=df["Class"].mean(), line_dash="dash", line_color="gray",
                       annotation_text="baseline")
        fig.update_layout(height=380, margin=dict(t=10), xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Logistic Regression vs Random Forest")

    comparison = {}
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        comparison[name] = {
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        }
    comp_df = pd.DataFrame(comparison).T.round(4)

    cc1, cc2 = st.columns([1, 1.4])
    with cc1:
        st.dataframe(comp_df.style.background_gradient(cmap="Greens"), use_container_width=True)
    with cc2:
        fig = px.bar(comp_df.reset_index().melt(id_vars="index"),
                      x="index", y="value", color="variable", barmode="group",
                      labels={"index": "Model", "value": "Score", "variable": "Metric"})
        fig.update_layout(height=320, margin=dict(t=10), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: Feature insights ----
with tab3:
    if model_choice == "Random Forest":
        st.subheader(f"Feature Importance — {model_choice}")
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig = px.bar(importances[::-1], orientation="h", color=importances[::-1].values,
                      color_continuous_scale="Viridis")
        fig.update_layout(height=500, margin=dict(t=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(f"Coefficient Magnitude — {model_choice}")
        coefs = pd.Series(log_reg.coef_[0], index=feature_cols)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)
        fig = px.bar(top[::-1], orientation="h", color=top[::-1].values,
                      color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=500, margin=dict(t=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap (top fraud-correlated features)")
    top_features = df.drop(columns=["Hour", "Amount_scaled"]).corr()["Class"].abs().sort_values(ascending=False).head(10).index
    heat = df[top_features].corr()
    fig = px.imshow(heat, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=500, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 4: Data explorer ----
with tab4:
    st.subheader("Flagged Transactions (at current threshold)")

    results_df = X_test.copy()
    results_df["Actual"] = y_test.values
    results_df["Fraud Probability"] = y_proba
    results_df["Predicted"] = y_pred

    only_fraud_pred = st.checkbox("Show only predicted fraud", value=True)
    view_df = results_df[results_df["Predicted"] == 1] if only_fraud_pred else results_df

    st.dataframe(
        view_df[["Fraud Probability", "Actual", "Predicted"]]
        .sort_values("Fraud Probability", ascending=False)
        .head(200)
        .style.background_gradient(subset=["Fraud Probability"], cmap="Reds"),
        use_container_width=True,
        height=450,
    )

    st.caption(f"Showing top 200 of {len(view_df):,} matching rows from the test set.")
