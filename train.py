"""Compare classifiers on chronological splits and export the chosen pipeline."""
import json
from io import BytesIO

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_source import ROOT, load_bytes

TARGET = "Room_Occupancy_Count"
LABELS = [0, 1, 2, 3]
FEATURES = {
    "minimal": ["S1_Light", "S5_CO2"],
    "ambient": [f"S{i}_Light" for i in range(1, 5)] + ["S5_CO2"],
    "full": [f"S{i}_{kind}" for kind in ("Temp", "Light", "Sound")
             for i in range(1, 5)] + ["S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"],
}
TRAIN_END = pd.Timestamp("2017-12-23")
TEST_START = pd.Timestamp("2018-01-10")
SEED = 42


def load_data():
    data = pd.read_csv(BytesIO(load_bytes()))
    data["timestamp"] = pd.to_datetime(data.Date + " " + data.Time)
    return data.sort_values("timestamp").reset_index(drop=True)


def split(data):
    return (
        data[data.timestamp < TRAIN_END],
        data[(data.timestamp >= TRAIN_END) & (data.timestamp < TEST_START)],
        data[data.timestamp >= TEST_START],
    )


def estimator_grid():
    yield "logistic", {}, make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    for depth in (4, 8, None):
        yield "forest", {"max_depth": depth}, RandomForestClassifier(
            n_estimators=100, max_depth=depth, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=1,
        )


def score(actual, predicted):
    supported = sorted(actual.unique())
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=supported,
                                 average="macro", zero_division=0)),
        "macro_f1_all_labels": float(f1_score(actual, predicted, labels=LABELS,
                                            average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def support(frame):
    return {str(label): int((frame[TARGET] == label).sum()) for label in LABELS}


def main():
    train, valid, test = split(load_data())
    candidates = []
    fitted = {}
    for group, columns in FEATURES.items():
        for name, params, model in estimator_grid():
            key = f"{group}/{name}/{params.get('max_depth', '-')}"
            model.fit(train[columns], train[TARGET])
            candidates.append({"id": key, "features": group, "model": name,
                               "parameters": params,
                               **score(valid[TARGET], model.predict(valid[columns]))})
            fitted[key] = model

    best = {group: max((row for row in candidates if row["features"] == group),
                      key=lambda row: row["macro_f1"]) for group in FEATURES}
    # Choose the smallest sensor set within 0.03 macro-F1 of the best candidate.
    threshold = max(row["macro_f1"] for row in best.values()) - 0.03
    selected = next(best[group] for group in FEATURES if best[group]["macro_f1"] >= threshold)
    columns = FEATURES[selected["features"]]
    development = pd.concat([train, valid])
    model = fitted[selected["id"]].fit(development[columns], development[TARGET])

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[columns], train[TARGET])
    validation_baseline = score(valid[TARGET], baseline.predict(valid[columns]))
    baseline.fit(development[columns], development[TARGET])
    report = {
        "split_support": {name: support(frame) for name, frame in
                          zip(("train", "validation", "test"), (train, valid, test))},
        "labels": LABELS,
        "validation_baseline": validation_baseline,
        "candidates": candidates,
        "best_by_sensor_set": best,
        "selected": selected,
        "selected_columns": columns,
        "test": score(test[TARGET], model.predict(test[columns])),
        "test_baseline": score(test[TARGET], baseline.predict(test[columns])),
    }
    directory = ROOT / "artifacts"
    directory.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": columns, "id": selected["id"]},
                directory / "model.joblib", compress=3)
    (directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("selected", "test", "test_baseline")},
                     indent=2))


if __name__ == "__main__":
    main()
