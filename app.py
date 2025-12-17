import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Space Debris Risk Classifier",
    page_icon="🛰",
    layout="wide"
)

st.title("🛰 Space Debris Risk Classifier")
st.markdown(
    "Upload **any space-related dataset** and understand whether space is "
    "**Low Risk, Medium Risk, or High Risk** 🌍"
)

# ============================================================
# SIDEBAR – UPLOAD CSVs
# ============================================================
st.sidebar.header("📁 Upload Space Data")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬅️ Please upload at least one CSV file to begin")
    st.stop()

dfs = []
for f in uploaded_files:
    temp = pd.read_csv(f)
    temp["SOURCE_FILE"] = f.name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# ============================================================
# AUTO-DETECT RISK COLUMN
# ============================================================
possible_targets = [
    "risk_level", "risk", "collision_risk",
    "danger_level", "threat"
]

risk_column = None
for col in df.columns:
    if col.lower() in possible_targets:
        risk_column = col
        break

if risk_column is None:
    st.error(
        "❌ No risk column found.\n\n"
        "CSV must contain one of:\n"
        "`risk_level, risk, collision_risk, danger_level, threat`"
    )
    st.stop()

# ============================================================
# DATA OVERVIEW
# ============================================================
st.header("📊 Dataset Overview")
st.dataframe(df.head())

c1, c2, c3 = st.columns(3)
c1.metric("Objects Analyzed", len(df))
c2.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
c3.metric("Risk Categories", df[risk_column].nunique())

# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================
X = df.drop(columns=[risk_column])
X = X.select_dtypes(include=[np.number])

if X.shape[1] < 2:
    st.error("❌ Dataset must contain at least 2 numeric columns")
    st.stop()

y = df[risk_column].astype(str)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ------------------------------------------------------------
# ORDERED RISK LABELS (Low, Medium, High)
# ------------------------------------------------------------
risk_label_map = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk"
}

ordered_labels = []
ordered_encoded = []

for lbl in ["low", "medium", "high"]:
    for cls in encoder.classes_:
        if cls.lower() == lbl:
            ordered_labels.append(risk_label_map[lbl])
            ordered_encoded.append(encoder.transform([cls])[0])

# ============================================================
# SCALING + PCA
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

st.header("📉 PCA – Data Simplification")
st.bar_chart(pd.Series(pca.explained_variance_ratio_))

# ============================================================
# PCA CORRELATION HEATMAP
# ============================================================
st.subheader("🔗 PCA Correlation Heatmap")

corr = X_pca_df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================
st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Data (%)", 10, 40, 20, 5)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca_df,
    y_encoded,
    test_size=test_size / 100,
    random_state=42
)

# ============================================================
# TRAIN RANDOM FOREST
# ============================================================
st.header("🧠 Train Random Forest Model")

if st.button("🚀 Train Random Forest"):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.success("✅ Random Forest model trained successfully")

# ============================================================
# MODEL EVALUATION
# ============================================================
if "model" in st.session_state:
    model = st.session_state.model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.header("📊 Model Evaluation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (%)", f"{accuracy:.2f}")
    m2.metric("Precision", f"{precision:.2f}")
    m3.metric("Recall", f"{recall:.2f}")
    m4.metric("F1 Score", f"{f1:.2f}")

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    st.subheader("📄 Classification Report (Low / Medium / High Risk)")
    st.text(
        classification_report(
            y_test,
            y_pred,
            labels=ordered_encoded,
            target_names=ordered_labels
        )
    )

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    st.subheader("📌 Confusion Matrix (Risk Levels)")
    cm = confusion_matrix(y_test, y_pred, labels=ordered_encoded)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted Risk")
    ax_cm.set_ylabel("Actual Risk")
    st.pyplot(fig_cm)

    # -----------------------------
    # OVERALL RISK DECISION
    # -----------------------------
    high_risk_label = ordered_encoded[-1]
    high_risk_percent = (np.sum(y_pred == high_risk_label) / len(y_pred)) * 100

    if high_risk_percent < 30:
        overall_risk = "🟢 LOW RISK"
    elif high_risk_percent <= 60:
        overall_risk = "🟡 MEDIUM RISK"
    else:
        overall_risk = "🔴 HIGH RISK"

    st.subheader("🚦 Final Space Collision Risk")
    st.markdown(f"## {overall_risk}")

# ============================================================
# PCA PROJECTION
# ============================================================
st.header("🌍 PCA Projection (PC1 vs PC2)")

if X_pca_df.shape[1] >= 2:
    viz_df = X_pca_df.copy()
    viz_df["Risk"] = y.astype(str)

    fig = px.scatter(
        viz_df,
        x="PC1",
        y="PC2",
        color="Risk",
        title="Low / Medium / High Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ROW EXPLORER (VIEW ONLY)
# ============================================================
st.header("🔎 Row Explorer (View Only)")

row_index = st.number_input(
    "Select row number to view",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

st.dataframe(df.iloc[[row_index]])

# ============================================================
# DOWNLOAD ANALYSIS REPORT
# ============================================================
st.header("📄 Download Analysis Report")

report_df = pd.DataFrame({
    "Total Objects Analyzed": [len(df)],
    "Numeric Features Used": [X.shape[1]],
    "PCA Components Used": [X_pca_df.shape[1]],
    "Accuracy (%)": [round(accuracy * 100, 2)],
    "Precision": [round(precision, 2)],
    "Recall": [round(recall, 2)],
    "F1 Score": [round(f1, 2)],
    "High Risk Percentage": [round(high_risk_percent, 2)],
    "Overall Collision Risk": [overall_risk],
    "Risk Decision Rule": ["<30% Low | 30–60% Medium | >60% High"]
})

st.dataframe(report_df)

csv_data = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Report (CSV)",
    data=csv_data,
    file_name="space_collision_risk_report.csv",
    mime="text/csv"
)