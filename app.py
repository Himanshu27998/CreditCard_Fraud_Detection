import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import altair as alt
import base64

# ----------------------- Page & Theming -----------------------
st.set_page_config(
    page_title="💳 HDFC Credit Card Fraud Prediction",
    page_icon="💳",
    layout="wide"
)

def _img_to_base64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

# Try to load brand assets if available
HDFC_LOGO_PATH = r"C:\Users\Admin\Downloads\hdfc_logo.png"
HDFC_BG_PATH = r"C:\Users\Admin\Downloads\hdfc_bg.jpg"
b64_logo = _img_to_base64(r"C:\Users\Admin\Downloads\hdfc_logo.png")
b64_bg = _img_to_base64(r"C:\Users\Admin\Downloads\hdfc_bg.jpg")

bg_css = f'background-image: url("data:image/jpg;base64,{b64_bg}");' if b64_bg else "background: radial-gradient(1200px 600px at 20% -10%, #e6eefc 0%, transparent 60%), linear-gradient(180deg, #f7f9ff 0%, #ffffff 60%);"

# Minimal CSS polish (cards, header accent, compact table option)
CUSTOM_CSS = f"""
<style>
/* App-wide text smoothing */
html, body, [class*="css"]  {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }}

body {{
  {bg_css}
  background-attachment: fixed;
  background-repeat: no-repeat;
  background-size: cover;
}}

.header-accent {{ 
  padding: 14px 18px; border-radius: 14px; color: white; 
  display: inline-flex; align-items: center; gap: 12px; font-weight: 600;
  background: #004080;  /* HDFC blue */
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}
.header-accent img {{ border-radius: 6px; background: #fff; padding: 4px; }}
.kpi {{
  border: 1px solid rgba(0,0,0,0.06);
  padding: 14px 16px; border-radius: 16px; background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.kpi .label {{ font-size: 12px; color: #666; margin-bottom: 6px; }}
.kpi .value {{ font-size: 22px; font-weight: 700; }}
.small {{ font-size: 12px; color:#777; }}
hr.soft {{ border: none; border-top: 1px solid rgba(0,0,0,0.06); margin: 12px 0 6px; }}
.footer-note {{ color:#7a7a7a; font-size:12px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------- Artifacts -----------------------
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

def preprocess_numpy(df, feature_names, med, mu, sigma):
    X = df[feature_names].to_numpy(dtype=float)
    X[np.isinf(X)] = np.nan
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    X_scaled = (X - mu) / sigma_safe
    return X_scaled

def predict_proba(X_scaled, coef, intercept):
    logits = X_scaled @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))

def fraud_vs_not_chart(preds: np.ndarray, accent="#97144D"):
    labels = np.where(preds == 1, "🚨 Fraud", "✅ Not Fraud")
    counts = pd.Series(labels).value_counts().rename_axis("Prediction").reset_index(name="Count")
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
                scale=alt.Scale(domain=["🚨 Fraud", "✅ Not Fraud"], range=["#d62728", accent]),
                legend=None
            ),
            tooltip=["Prediction:N", "Count:Q"]
        )
        .properties(height=300)
    )
    return chart

def proba_hist_chart(proba: np.ndarray, bins=30, accent="#97144D"):
    dfh = pd.DataFrame({"probability": proba})
    chart = (
        alt.Chart(dfh)
        .transform_bin("bin", "probability", bin=alt.Bin(maxbins=bins))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("bin:Q", title="Fraud Probability (binned)"),
            y=alt.Y("count():Q", title="Records"),
            tooltip=[alt.Tooltip("count():Q", title="Count")]
        )
        .properties(height=300)
        .configure_mark(color=accent)
    )
    return chart

# ----------------------- Sidebar Controls -----------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    accent = st.color_picker("Accent color", value="#004080")  # HDFC blue default
    threshold = st.slider("Decision threshold (≥ = Fraud)", 0.0, 1.0, 0.50, 0.01)
    decimals = st.slider("Probability decimals", 2, 6, 2, 1)
    compact = st.toggle("Compact table", value=True)
    st.markdown("---")
    st.markdown("### ℹ️ Help")
    st.caption(
        "Paste values for the required features in the main panel. "
        "Use commas / tabs / spaces. Click **Predict** to score."
    )

# ----------------------- Header -----------------------
feature_names, med, mu, sigma, coef, intercept = load_artifacts()

if b64_logo:
    header_html = f"""
    <div class="header-accent" style="background:{accent};">
        <img src="data:image/png;base64,{b64_logo}" width="110">
        <span>💳 HDFC Credit Card Fraud Prediction</span>
    </div>
    """
else:
    header_html = f"""
    <div class="header-accent" style="background:{accent};">
        💳 HDFC Credit Card Fraud Prediction
    </div>
    """
st.markdown(header_html, unsafe_allow_html=True)
st.write("A secure, branded paste-only scoring tool for internal demos & reviews. (No sklearn at runtime.)")

with st.expander("Required feature order (copy for Excel exports)"):
    st.code(", ".join(feature_names), language="text")

# ----------------------- Input -----------------------
example = ",".join(["0"] * len(feature_names))
text = st.text_area(
    "Paste rows (one per line). Commas, tabs, or spaces between numbers are accepted.",
    value=example,
    height=140
)

colL, colR = st.columns([1,1])
with colL:
    predict_clicked = st.button("🔮 Predict", type="primary")
with colR:
    template = ", ".join(["0"] * len(feature_names))
    st.download_button(
        "⬇️ Download input template (.txt)",
        data=template,
        file_name="fraud_input_template.txt",
        mime="text/plain"
    )

# ----------------------- Predict Flow -----------------------
if predict_clicked:
    try:
        df = parse_pasted_text(text, feature_names)

        X_scaled = preprocess_numpy(df, feature_names, med, mu, sigma)
        proba = predict_proba(X_scaled, coef, intercept)
        preds = (proba >= threshold).astype(int)

        out = df.copy()
        out["fraud_probability"] = np.round(proba, decimals)
        out["prediction"] = preds.astype(int)

        # KPIs
        tot = len(out)
        frauds = int((out["prediction"] == 1).sum())
        avgp = float(out["fraud_probability"].mean()) if tot else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="kpi"><div class="label">Records Scored</div>'
                        f'<div class="value">{tot:,}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="kpi"><div class="label">Predicted Fraud</div>'
                        f'<div class="value">{frauds:,}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="kpi"><div class="label">Avg Fraud Probability</div>'
                        f'<div class="value">{avgp:.{decimals}f}</div></div>', unsafe_allow_html=True)

        st.markdown("#### Results")
        st.dataframe(
            out if not compact else out.style.set_properties(**{"font-size": "12px"}),
            use_container_width=True,
            hide_index=True
        )

        # Charts row
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Predictions Breakdown**")
            st.altair_chart(fraud_vs_not_chart(preds, accent), use_container_width=True)
        with c2:
            st.markdown("**Fraud Probability Distribution**")
            st.altair_chart(proba_hist_chart(proba, bins=30, accent=accent), use_container_width=True)

        # Download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download scored results (CSV)",
            data=csv,
            file_name="fraud_scored_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info(
            "Tip: Ensure each row has exactly the required number of values "
            "and in the same order as shown above."
        )

# ----------------------- Footer / About -----------------------
st.markdown("hr", unsafe_allow_html=True)
with st.expander("About this app"):
    st.write(
        "- **Paste-only**: No file uploads or sklearn needed at runtime\n"
        "- **Preprocessing**: median imputation, z-score scaling (using saved stats)\n"
        "- **Model**: logistic regression (coef & intercept loaded from JSON)\n"
        "- **Threshold**: configurable from the sidebar"
    )
st.markdown('<div class="footer-note">© HDFC Bank • Internal Fraud Detection Demo</div>', unsafe_allow_html=True)
