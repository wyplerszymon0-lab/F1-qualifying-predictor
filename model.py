import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import (
    generate_qualifying_data, DRIVERS,
    TEAM_QUALI_PACE, CIRCUIT_QUALI_TRAITS,
    DRIVER_CIRCUIT_AFFINITY,
)

FEATURES = [
    "quali_skill", "team_pace", "traction", "braking",
    "aero", "track_evo", "temperature", "humidity",
    "is_damp", "affinity",
]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df["pole"]
    return X, y


def train_model(df: pd.DataFrame) -> tuple:
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


def predict_qualifying(model, circuit: str, temperature: float = 25.0,
                       humidity: float = 50.0, is_damp: int = 0) -> pd.DataFrame:
    if circuit not in CIRCUIT_QUALI_TRAITS:
        raise ValueError(f"Unknown circuit. Available: {list(CIRCUIT_QUALI_TRAITS.keys())}")

    traits = CIRCUIT_QUALI_TRAITS[circuit]
    rows   = []

    for code, info in DRIVERS.items():
        rows.append({
            "driver":      code,
            "name":        info["name"],
            "team":        info["team"],
            "quali_skill": info["quali_skill"],
            "team_pace":   TEAM_QUALI_PACE[info["team"]],
            "traction":    traits["traction"],
            "braking":     traits["braking"],
            "aero":        traits["aero"],
            "track_evo":   traits["track_evo"],
            "temperature": temperature,
            "humidity":    humidity,
            "is_damp":     is_damp,
            "affinity":    DRIVER_CIRCUIT_AFFINITY.get((code, circuit), 0),
        })

    df    = pd.DataFrame(rows)
    X     = df[FEATURES]
    probs = model.predict_proba(X)[:, 1]

    df["pole_probability"] = probs / probs.sum()
    df["pole_pct"]         = (df["pole_probability"] * 100).round(1)

    return df.sort_values("pole_probability", ascending=False).reset_index(drop=True)


def feature_importance(model) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
