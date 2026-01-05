import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
test_df = pd.read_csv("../data/raw/test.csv")

X = test_df.drop(columns=["Activity", "subject"])
y_true = test_df["Activity"]

# Load model + preprocessing
model = joblib.load("../model/extra_trees_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
selected_idx = joblib.load("../model/selected_features.pkl")

# Transform
X_scaled = scaler.transform(X)
X_selected = X_scaled[:, selected_idx]

# Predict
y_pred_enc = model.predict(X_selected)

label_map = {
    0: "LAYING",
    1: "SITTING",
    2: "STANDING",
    3: "WALKING",
    4: "WALKING_DOWNSTAIRS",
    5: "WALKING_UPSTAIRS"
}

y_pred = [label_map[p] for p in y_pred_enc]

# Prediction results
results_df = pd.DataFrame({
    "True_Activity": y_true,
    "Predicted_Activity": y_pred
})
results_df.to_csv("../powerbi/predictions.csv", index=False)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("../powerbi/classification_report.csv")

#  Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(label_map.values()))
cm_df = pd.DataFrame(
    cm,
    index=list(label_map.values()),
    columns=list(label_map.values())
)
cm_df.to_csv("../powerbi/confusion_matrix.csv")

print("✅ Power BI files exported successfully")
