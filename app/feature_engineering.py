import pandas as pd

def engineer_features(X_raw, scaler):
    """ EXACT feature engineering used in training """
    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        columns=X_raw.columns
    )

    # Statistical features
    X_stats = pd.DataFrame({
        "row_mean": X_raw.mean(axis=1),
        "row_std": X_raw.std(axis=1),
        "row_max": X_raw.max(axis=1),
        "row_min": X_raw.min(axis=1)
    })

    # Energy features
    energy_cols = [c for c in X_raw.columns if "energy()" in c]
    X_energy = pd.DataFrame({
        "energy_mean": X_raw[energy_cols].mean(axis=1),
        "energy_std": X_raw[energy_cols].std(axis=1)
    })

    # Angle features
    angle_cols = [c for c in X_raw.columns if c.startswith("angle")]
    X_angles = X_raw[angle_cols]

    # Final matrix
    X_engineered = pd.concat(
        [X_scaled, X_stats, X_energy, X_angles],
        axis=1
    )

    return X_engineered