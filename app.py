import streamlit as st
import pandas as pd
import numpy as np
import json, re, altair as alt
from datetime import datetime

st.set_page_config(page_title="💳 Credit Card Fraud Prediction", page_icon="💳", layout="wide")

# ===== Presentation settings =====
ACCENT = "#004080"   # header accent (HDFC blue style)
THRESHOLD = 0.50
DECIMALS = 2

# ===== Global CSS =====
def inject_css():
    css = f"""
    <style>
      .stApp {{
        background: linear-gradient(135deg, #0b1220 0%, #1c2233 100%);
        color: #f5f5f5;
      }}

      .block-container {{
        background: rgba(245,245,245,0.97);
        border-radius: 18px;
        padding: 28px 28px 36px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        max-width: 1200px;
      }}

      .brandbar {{
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(90deg, #003366 0%, #004080 100%);
        color:white !important;
        border-radius:14px;
        padding:18px 24px;
        margin-bottom:18px;
        font-family: 'Segoe UI', sans-serif;
        font-weight:800;
        font-size:26px;
        letter-spacing:0.5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
        box-shadow:0 4px 15px rgba(0,0,0,0.4);
      }}

      h4, h5, .stMarkdown h4, .stMarkdown h5 {{
        font-family: 'Segoe UI', sans-serif;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: {ACCENT} !important;
      }}

      thead tr th {{
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #000 !important;
        background-color: #f0f0f0 !important;
      }}
      tbody tr td {{
        font-size: 13px !important;
        color: #111 !important;
      }}

      .badge {{
        font-size: 13px; font-weight: 700; padding: 2px 8px;
        border-radius: 8px; color:#fff;
      }}
      .badge.low  {{ background:#2ca02c; }}
      .badge.med  {{ background:#ff7f0e; }}
      .badge.high {{ background:#d62728; }}

      .kpi {{
        border-radius:16px;
        padding:14px 16px;
        box-shadow:0 2px 10px rgba(0,0,0,0.15);
        color:#000; background:#ffffff;
      }}
      .kpi .label {{ font-size:12px; color:#555; margin-bottom:6px; }}
      .kpi .value {{ font-size:22px; font-weight:700; }}

      .kpi-ok    {{ background:#2ca02c !important; color:#fff !important; }}
      .kpi-alert {{ background:#d62728 !important; color:#fff !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ===== Load artifacts =====
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

# ===== Functions =====
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
    if any(len(r) != len(expected_cols) for r in rows):
        raise ValueError(f"Each row must have {len(expected_cols)} values in order: {', '.join(expected_cols)}")
    return pd.DataFrame([[float(x) for x in r] for r in rows], columns=expected_cols)

def preprocess_numpy(df, feature_names, med, mu, sigma):
    X = df[feature_names].to_numpy(dtype=float)
    X[np.isinf(X)] = np.nan
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma_safe

def predict_proba(X_scaled, coef, intercept):
    logits = X_scaled @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))

def risk_tier(p):
    if p >= 0.80: return "High", "badge high", "Auto-block & verify"
    elif p >= 0.50: return "Medium", "badge med", "Step-up auth; monitor"
    else: return "Low", "badge low", "Allow & monitor"

def fraud_vs_not_chart(preds: np.ndarray):
    labels = np.where(preds == 1, "🚨 Fraud", "✅ Not Fraud")
    counts = pd.Series(labels).value_counts().rename_axis("Prediction").reset_index(name="Count")
    wanted = pd.DataFrame({"Prediction": ["🚨 Fraud","✅ Not Fraud"]})
    counts = wanted.merge(counts, on="Prediction", how="left").fillna({"Count":0})
    counts["Count"] = counts["Count"].astype(int)

    return (
        alt.Chart(counts, background="transparent")
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X(
                "Prediction:N",
                title="",
                axis=alt.Axis(labelFontSize=14, labelFontWeight="bold")
            ),
            y=alt.Y("Count:Q", title="Records"),
            color=alt.Color(
                "Prediction:N",
                scale=alt.Scale(
                    domain=["🚨 Fraud", "✅ Not Fraud"],
                    range=["#d62728", "#2ca02c"]  # red for fraud, green for not fraud
                ),
                legend=None
            ),
            tooltip=["Prediction:N","Count:Q"]
        )
        .properties(height=300)
    )

# ===== UI =====
st.markdown(f"<div class='brandbar'>💳 HDFC Credit Card Fraud Prediction</div>", unsafe_allow_html=True)

feature_names, med, mu, sigma, coef, intercept = load_artifacts()

# ===== Feature order =====
st.markdown("<h4 style='font-family:Segoe UI; font-size:20px; font-weight:600; color:#004080;'>Required Feature Order</h4>", unsafe_allow_html=True)
st.code(", ".join(feature_names), language="text")

# ===== Input area =====
example = ",".join(["0"] * len(feature_names))
text = st.text_area(
    "",
    value=example,
    height=140,
    placeholder="Paste feature rows here (comma/tab/space separated, one row per line)…"
)

# Buttons
c1, c2 = st.columns([1,1])
with c1:
    predict_clicked = st.button("🔮 Predict", type="primary")
with c2:
    st.download_button("⬇ Download input template", data=example, file_name="fraud_input_template.txt", mime="text/plain")

# ===== Predict flow =====
if predict_clicked:
    try:
        df = parse_pasted_text(text, feature_names)
        X_scaled = preprocess_numpy(df, feature_names, med, mu, sigma)
        proba = predict_proba(X_scaled, coef, intercept)
        preds = (proba >= THRESHOLD).astype(int)

        out = df.copy()
        out["fraud_probability"] = np.round(proba, DECIMALS)
        out["prediction"] = preds.astype(int)
        tiers = [risk_tier(p) for p in proba]
        out["risk_tier"]   = [t[0] for t in tiers]
        out["next_action"] = [t[2] for t in tiers]

        # KPIs
        tot = len(out)
        frauds = int((out["prediction"] == 1).sum())
        avgp = float(out["fraud_probability"].mean()) if tot else 0.0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='kpi'><div class='label'>Records</div><div class='value'>{tot:,}</div></div>", unsafe_allow_html=True)
        with k2:
            fraud_class = "kpi-alert" if frauds > 0 else "kpi-ok"
            st.markdown(f"<div class='kpi {fraud_class}'><div class='label'>Frauds Flagged</div><div class='value'>{frauds:,}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi'><div class='label'>Avg Probability</div><div class='value'>{avgp:.{DECIMALS}f}</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi'><div class='label'>Decision Time</div><div class='value'>{datetime.now().strftime('%H:%M:%S')}</div></div>", unsafe_allow_html=True)

        # Results
        st.markdown("#### Results")
        show = out.copy()
        show["risk_badge"] = [f"<span class='{risk_tier(p)[1]}'>{risk_tier(p)[0]}</span>" for p in proba]
        st.write(show[["fraud_probability","prediction","risk_badge","next_action"]].to_html(escape=False, index=False), unsafe_allow_html=True)

        # Predictions Breakdown chart
        st.markdown("#### Predictions Breakdown")
        st.altair_chart(fraud_vs_not_chart(preds), use_container_width=True)

        # Export
        st.download_button("💾 Download results (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="fraud_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
