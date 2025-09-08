import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import altair as alt

st.set_page_config(page_title="💳 Credit Card Fraud Prediction", page_icon="💳", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open("preproc_stats.json", "r") as f:
        pre = json.load(f)
    with open("model_params.json", "r") as f:
        model = json.load(f)
    med = np.array(pre["medians"], dtype=float)
    mu = np.array(pre["means"], dtype=float)
    sigma = np.array(pre["stds"], dtype=float)
    coef = np.array(model["coef"], dtype=float).reshape(-1)
    intercept = float(model["intercept"])
    return feature_names, med, mu, sigma, coef, intercept

# ---------- Parse pasted text ----------
def parse_pasted_text(text: str, expected_cols):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if "," in ln:
            parts = [s.strip() for s in ln.split(",") if s.strip() != ""]
        else:
            ln = ln.replace("\t", " ")
            ln = re.sub(r"\s+", " ", ln)
            parts = [s.strip() for s in ln.split(" ") if s.strip() != ""]
        rows.append(parts)
    if not rows:
        raise ValueError("No values detected. Paste at least one row.")
    n_expected = len(expected_cols)
    for i, r in enumerate(rows, start=1):
        if len(r) != n_expected:
            raise ValueError(
                f"Row {i} has {len(r)} values, expected {n_expected}.\n"
                f"Expected order: {', '.join(expected_cols)}"
            )
    data = [[float(x) for x in r] for r in rows]
    return pd.DataFrame(data, columns=expected_cols)

# ---------- Preprocess ----------
def preprocess_numpy(df, feature_names, med, mu, sigma):
    X = df[feature_names].to_numpy(dtype=float)
    X[np.isinf(X)] = np.nan
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    X_scaled = (X - mu) / sigma_safe
    return X_scaled

# ---------- Logistic probability ----------
def predict_proba(X_scaled, coef, intercept):
    logits = X_scaled @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))

# ---------- Chart: simple Fraud vs Not Fraud counts ----------
def fraud_vs_not_chart(preds: np.ndarray):
    labels = np.where(preds == 1, "🚨 Fraud", "✅ Not Fraud")
    counts = pd.Series(labels).value_counts().rename_axis("Prediction").reset_index(name="Count")
    # ensure both categories always appear (even if zero)
    wanted = pd.DataFrame({"Prediction": ["🚨 Fraud", "✅ Not Fraud"]})
    counts = wanted.merge(counts, on="Prediction", how="left").fillna({"Count": 0})
    counts["Count"] = counts["Count"].astype(int)

    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Prediction:N", title=""),
            y=alt.Y("Count:Q", title="Number of Records"),
            color=alt.Color(
                "Prediction:N",
                scale=alt.Scale(domain=["🚨 Fraud", "✅ Not Fraud"], range=["#d62728", "#2ca02c"]),
                legend=None
            ),
            tooltip=["Prediction:N", "Count:Q"]
        )
        .properties(height=300)
    )
    return chart

# ---------- UI ----------
def main():
    st.title("💳 Credit Card Fraud Prediction")
    st.write("Paste values for the features below (same order). No file upload or CSV needed.")

    feature_names, med, mu, sigma, coef, intercept = load_artifacts()

    with st.expander("Required feature order"):
        st.code(", ".join(feature_names), language="text")

    example = ",".join(["0"] * len(feature_names))
    text = st.text_area(
        "Paste rows (one per line). Use commas, tabs, or spaces between numbers.",
        value=example,
        height=140
    )

    if st.button("Predict"):
        try:
            df = parse_pasted_text(text, feature_names)
            threshold = 0.50  # fixed internal threshold

            X_scaled = preprocess_numpy(df, feature_names, med, mu, sigma)
            proba = predict_proba(X_scaled, coef, intercept)
            preds = (proba >= threshold).astype(int)

            out = df.copy()
            out["fraud_probability"] = proba
            out["prediction"] = preds

            # Results table
            st.subheader("Results")
            st.dataframe(
                out,
                use_container_width=True,
                column_config={
                    "fraud_probability": st.column_config.ProgressColumn(
                        "Fraud Probability",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    "prediction": st.column_config.NumberColumn(
                        "Prediction (1=Fraud, 0=Not Fraud)"
                    ),
                },
                hide_index=True,
            )

            # Single visualization: Fraud vs Not Fraud
            st.altair_chart(fraud_vs_not_chart(preds), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure each line has exactly the required number of values in the correct order.")

    st.caption("Tip: Copy cells for V1..V28 and Amount from Excel (no headers) and paste here.")

if __name__ == "__main__":
    main()
