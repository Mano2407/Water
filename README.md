# My Water Tracker (Streamlit)

Track daily water/drink intake, see progress toward a goal, and view a daily report.

## Run
```
pip install -r requirements.txt
python train.py          # optional: retrains model.joblib + metrics.json
streamlit run app.py
```

## Files
- `app.py` – Streamlit UI (tracker + daily report)
- `pipeline.py` – feature pipeline and goal prediction
- `train.py` – trains the recommended-goal model
- `styles.css` – custom styling, `.streamlit/config.toml` – theme
- `data/` – training data and `water_log.csv` (created automatically)
- `model.joblib`, `metrics.json` – trained model and its metrics

Note: on Streamlit Cloud the log file is shared and resets on redeploy.
