# Pause & Protect — Demo

## Setup

1. `pip install -r requirements.txt`
2. `python app/train_model.py` — trains the risk model on synthetic data and prints a classification report.
3. `streamlit run app/app.py`

No `.env` or API key is required to run the demo — without one, warnings are generated from a template built from the same risk factors the model surfaces, which is a fully legitimate way to run this (no cost, nothing to configure).

**Optional:** if you want the warning text generated live by Claude instead of the template, create a `.env` file in this folder with `ANTHROPIC_API_KEY=your-key-here`. If the key is missing, invalid, or the API call fails for any reason (no network, rate limit, timeout), it falls back to the template automatically — the live demo never breaks either way.

## Files

- `train_model.py` — generates synthetic transfer data and trains the risk-scoring model (`models/risk_model.pkl`).
- `model.py` — turns a transfer into features and a risk score/level for the persona.
- `llm_explainer.py` — turns the risk score into a plain-language warning via Claude, with the fallback described above.
- `app.py` — the Streamlit demo itself.
