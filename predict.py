"""
predict.py
==========
Loads the trained XGBoost pipeline (best_xgboost.pkl, saved by the
notebook's `joblib.dump(grid_xgb.best_estimator_, "best_xgboost.pkl")`
cell) and predicts Extrovert / Introvert for new, raw input data.

The saved pipeline already contains the preprocessing step
(StandardScaler on numeric features + OneHotEncoder on categorical
features), so this script only needs to:
  1. Accept the 7 raw fields the original dataset had.
  2. Re-create the engineered features, using the exact same formulas
     as the notebook's feature-engineering cell.
  3. Feed the resulting row into the pipeline's .predict() / .predict_proba().

Usage
-----
As a library:
    from predict import predict_personality
    result = predict_personality({
        "Time_spent_Alone": 4.0,
        "Stage_fear": "No",
        "Social_event_attendance": 4.0,
        "Going_outside": 6.0,
        "Drained_after_socializing": "No",
        "Friends_circle_size": 13.0,
        "Post_frequency": 5.0,
    })
    print(result)   # {'prediction': 'Extrovert', ...}

As a CLI:
    python predict.py --time_spent_alone 4 --stage_fear No \\
        --social_event_attendance 4 --going_outside 6 \\
        --drained_after_socializing No --friends_circle_size 13 \\
        --post_frequency 5
"""

import argparse

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "best_xgboost.pkl"

LABEL_MAP = {0: "Extrovert", 1: "Introvert"}

RAW_REQUIRED_FIELDS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

# Column order the pipeline's ColumnTransformer was fit on. Must match the
# notebook's X_full (i.e. df_imputed.drop(columns=['Personality', 'target'])).
MODEL_INPUT_COLUMNS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
    "social_activity_score",
    "isolation_score",
    "social_fatigue_index",
    "social_confidence",
    "online_vs_offline_ratio",
    "behavior_variance",
    "alone_vs_friends",
    "social_outing_density",
]


def _engineer_features(row: dict) -> dict:
    """Re-create the same engineered features used in the notebook."""
    row = dict(row)  # don't mutate caller's dict

    yes_no_map = {"Yes": 1, "No": 0}

    row["social_activity_score"] = (
        row["Social_event_attendance"] + row["Going_outside"] + row["Post_frequency"]
    )

    row["isolation_score"] = row["Time_spent_Alone"] - row["Social_event_attendance"]

    row["social_fatigue_index"] = (
        yes_no_map[row["Drained_after_socializing"]] * yes_no_map[row["Stage_fear"]]
    )

    row["social_confidence"] = row["Friends_circle_size"] * row["Social_event_attendance"]

    row["online_vs_offline_ratio"] = row["Post_frequency"] / (row["Going_outside"] + 1)

    behavior_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    behavior_values = [row[c] for c in behavior_cols]
    # pandas .var(axis=1) on a single row uses ddof=1 (sample variance) by default
    row["behavior_variance"] = float(np.var(behavior_values, ddof=1))

    row["alone_vs_friends"] = row["Time_spent_Alone"] / (row["Friends_circle_size"] + 1)
    row["social_outing_density"] = row["Social_event_attendance"] * row["Going_outside"]

    return row


def _validate_input(raw_input: dict) -> None:
    missing = [f for f in RAW_REQUIRED_FIELDS if f not in raw_input]
    if missing:
        raise ValueError(f"Missing required input field(s): {missing}")

    for field in ("Stage_fear", "Drained_after_socializing"):
        if raw_input[field] not in ("Yes", "No"):
            raise ValueError(
                f"'{field}' must be 'Yes' or 'No', got: {raw_input[field]!r}"
            )


def load_model(model_path: str = MODEL_PATH):
    """Load the trained pipeline (preprocessor -> XGBoost)."""
    return joblib.load(model_path)


def predict_personality(raw_input: dict, model=None) -> dict:
    """
    Predict personality (Extrovert/Introvert) from raw survey-style input.

    Parameters
    ----------
    raw_input : dict
        Must contain the 7 raw fields:
        Time_spent_Alone, Stage_fear, Social_event_attendance,
        Going_outside, Drained_after_socializing, Friends_circle_size,
        Post_frequency.
    model : sklearn Pipeline, optional
        Pre-loaded model. If not provided, it is loaded from MODEL_PATH.

    Returns
    -------
    dict with keys: "prediction", "probability_extrovert",
    "probability_introvert".
    """
    _validate_input(raw_input)

    if model is None:
        model = load_model()

    engineered_row = _engineer_features(raw_input)
    X_new = pd.DataFrame([engineered_row], columns=MODEL_INPUT_COLUMNS)

    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    return {
        "prediction": LABEL_MAP[int(pred)],
        "probability_extrovert": float(proba[0]),
        "probability_introvert": float(proba[1]),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Extrovert/Introvert personality from survey inputs."
    )
    parser.add_argument("--time_spent_alone", type=float, required=True)
    parser.add_argument("--stage_fear", choices=["Yes", "No"], required=True)
    parser.add_argument("--social_event_attendance", type=float, required=True)
    parser.add_argument("--going_outside", type=float, required=True)
    parser.add_argument(
        "--drained_after_socializing", choices=["Yes", "No"], required=True
    )
    parser.add_argument("--friends_circle_size", type=float, required=True)
    parser.add_argument("--post_frequency", type=float, required=True)
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH, help="Path to the .pkl model"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    raw_input = {
        "Time_spent_Alone": args.time_spent_alone,
        "Stage_fear": args.stage_fear,
        "Social_event_attendance": args.social_event_attendance,
        "Going_outside": args.going_outside,
        "Drained_after_socializing": args.drained_after_socializing,
        "Friends_circle_size": args.friends_circle_size,
        "Post_frequency": args.post_frequency,
    }

    model = load_model(args.model_path)
    result = predict_personality(raw_input, model=model)

    print("\nInput:")
    for k, v in raw_input.items():
        print(f"  {k}: {v}")

    print("\nPrediction:", result["prediction"])
    print(f"  P(Extrovert) = {result['probability_extrovert']:.4f}")
    print(f"  P(Introvert) = {result['probability_introvert']:.4f}")


if __name__ == "__main__":
    main()