"""Train the daily-goal model. Run: python train.py"""
import json

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import BASE, FEATURES, MODEL_PATH, TARGET, build_pipeline, make_dataset


def main():
    df = make_dataset()
    (BASE / "data").mkdir(exist_ok=True)
    df.to_csv(BASE / "data" / "hydration_training.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    pipe = build_pipeline().fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "mae_ml": round(float(mean_absolute_error(y_test, pred)), 2),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    joblib.dump(pipe, MODEL_PATH)
    (BASE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model.joblib and metrics.json:", metrics)


if __name__ == "__main__":
    main()
