# Streamlit Fraud App — Paste-Only, No scikit-learn at Runtime

This app avoids scikit-learn version issues. It loads preprocessing stats and a trained logistic regression's parameters from JSON, and computes predictions with pure NumPy.

## Files
- app.py — Streamlit app (paste-only input)
- feature_names.json — feature order (e.g., V1..V28, Amount)
- preproc_stats.json — medians (for imputation), means and stds (for standardization)
- model_params.json — logistic regression coefficients and intercept
- requirements.txt — only streamlit, pandas, numpy

## Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Use
- Expand "Required feature order" to confirm columns.
- Paste rows (one per line). Commas/tabs/spaces allowed.
- Click Predict → see fraud_probability and prediction on screen.
