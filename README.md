# Listing Lab — Streamlit deployment bundle

This folder contains only the public app, dependencies, configuration, and 926 synthetic Perplexity Decider training labels. It excludes the five manual labels and the local benchmark results.

## Local preview

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Leave the OpenRouter model selection empty to test BGE and the two local supervised models without a key.

## Streamlit Community Cloud

1. Upload **the contents of this folder** to a GitHub repository, preserving `.streamlit/config.toml` and `public_data/teacher_labels.jsonl`.
2. Create a new app at `share.streamlit.io`, select that repository and `streamlit_app.py`, and choose Python 3.12.
3. In OpenRouter, create a dedicated API key and assign a **US$0.01 per day** spending guardrail to it.
4. After verifying the guardrail assignment, add the following to the app's **Secrets** settings:

```toml
OPENROUTER_API_KEY = "your-dedicated-key"
OPENROUTER_DAILY_GUARDRAIL_CONFIRMED = "true"
```

The app will not send OpenRouter requests until the confirmation flag is set. Never commit `secrets.toml` or the API key. Community Cloud can sleep after inactivity and its memory allocation varies, so test the BGE inference on the deployed app before sharing the URL widely.
