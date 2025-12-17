import pandas as pd
import joblib

def predict_from_csv(csv_file):
    model = joblib.load("../model/extra_trees_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    selected_idx = joblib.load("../model/selected_features.pkl")

    df = pd.read_csv(csv_file)
    X = df.drop(columns=["Activity", "subject"], errors="ignore")

    X_scaled = scaler.transform(X)
    X_selected = X_scaled[:, selected_idx]

    preds = model.predict(X_selected)

    label_map = {
        0: "LAYING",
        1: "SITTING",
        2: "STANDING",
        3: "WALKING",
        4: "WALKING_DOWNSTAIRS",
        5: "WALKING_UPSTAIRS"
    }

    return [label_map[int(p)] for p in preds]
