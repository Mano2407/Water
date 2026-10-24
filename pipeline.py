"""Feature pipeline + prediction helper for the recommended daily water goal."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
FEATURES = ["weight_kg", "age", "activity_level", "temp_c"]  # activity_level: 0 low, 1 medium, 2 high
TARGET = "goal_ml"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)),
    ])


def make_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "weight_kg": rng.uniform(35, 120, n).round(1),
        "age": rng.integers(10, 80, n),
        "activity_level": rng.integers(0, 3, n),
        "temp_c": rng.uniform(10, 42, n).round(1),
    })
    goal = (
        df.weight_kg * 33
        + df.activity_level * 350
        + np.clip(df.temp_c - 25, 0, None) * 30
        - np.clip(df.age - 55, 0, None) * 8
        + rng.normal(0, 80, n)
    )
    df[TARGET] = (np.clip(goal, 1200, 5000) / 50).round() * 50
    return df


def load_model():
    return joblib.load(MODEL_PATH)


def predict_goal_ml(weight_kg: float, age: int, activity_level: int, temp_c: float) -> int:
    x = pd.DataFrame([[weight_kg, age, activity_level, temp_c]], columns=FEATURES)
    ml = float(load_model().predict(x)[0])
    return int(round(ml / 50) * 50)
