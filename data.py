import numpy as np
import pandas as pd

DRIVERS = {
    "VER": {"name": "Max Verstappen",  "team": "Red Bull",    "skill": 0.98, "quali_skill": 0.97},
    "PER": {"name": "Sergio Perez",    "team": "Red Bull",    "skill": 0.82, "quali_skill": 0.79},
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari",     "skill": 0.91, "quali_skill": 0.94},
    "SAI": {"name": "Carlos Sainz",    "team": "Ferrari",     "skill": 0.88, "quali_skill": 0.87},
    "HAM": {"name": "Lewis Hamilton",  "team": "Mercedes",    "skill": 0.93, "quali_skill": 0.95},
    "RUS": {"name": "George Russell",  "team": "Mercedes",    "skill": 0.87, "quali_skill": 0.89},
    "NOR": {"name": "Lando Norris",    "team": "McLaren",     "skill": 0.90, "quali_skill": 0.91},
    "PIA": {"name": "Oscar Piastri",   "team": "McLaren",     "skill": 0.85, "quali_skill": 0.84},
    "ALO": {"name": "Fernando Alonso", "team": "Aston Martin","skill": 0.89, "quali_skill": 0.90},
    "STR": {"name": "Lance Stroll",    "team": "Aston Martin","skill": 0.75, "quali_skill": 0.72},
}

TEAM_QUALI_PACE = {
    "Red Bull":    0.96,
    "Ferrari":     0.90,
    "McLaren":     0.89,
    "Mercedes":    0.87,
    "Aston Martin":0.80,
}

CIRCUIT_QUALI_TRAITS = {
    "Bahrain":     {"traction": 0.7, "braking": 0.8, "aero": 0.6, "track_evo": 0.5},
    "Australia":   {"traction": 0.5, "braking": 0.7, "aero": 0.6, "track_evo": 0.4},
    "Japan":       {"traction": 0.4, "braking": 0.6, "aero": 0.9, "track_evo": 0.3},
    "Monaco":      {"traction": 0.9, "braking": 0.9, "aero": 0.7, "track_evo": 0.9},
    "Spain":       {"traction": 0.5, "braking": 0.7, "aero": 0.8, "track_evo": 0.5},
    "Britain":     {"traction": 0.4, "braking": 0.6, "aero": 0.8, "track_evo": 0.4},
    "Hungary":     {"traction": 0.7, "braking": 0.8, "aero": 0.9, "track_evo": 0.6},
    "Belgium":     {"traction": 0.3, "braking": 0.5, "aero": 0.7, "track_evo": 0.3},
    "Italy":       {"traction": 0.3, "braking": 0.8, "aero": 0.3, "track_evo": 0.4},
    "Singapore":   {"traction": 0.9, "braking": 0.9, "aero": 0.8, "track_evo": 0.9},
    "Abu Dhabi":   {"traction": 0.5, "braking": 0.7, "aero": 0.6, "track_evo": 0.5},
}

DRIVER_CIRCUIT_AFFINITY = {
    ("LEC", "Monaco"):   0.08,
    ("HAM", "Britain"):  0.07,
    ("VER", "Japan"):    0.06,
    ("ALO", "Monaco"):   0.05,
    ("HAM", "Hungary"):  0.05,
    ("VER", "Belgium"):  0.05,
}


def generate_qualifying_data(n_sessions: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []

    for session_id in range(n_sessions):
        circuit  = np.random.choice(list(CIRCUIT_QUALI_TRAITS.keys()))
        traits   = CIRCUIT_QUALI_TRAITS[circuit]
        temp     = np.random.uniform(15, 40)
        humidity = np.random.uniform(20, 90)
        is_damp  = 1 if np.random.random() < 0.1 else 0

        scores = {}
        for code, info in DRIVERS.items():
            team_pace    = TEAM_QUALI_PACE[info["team"]]
            driver_quali = info["quali_skill"]
            affinity     = DRIVER_CIRCUIT_AFFINITY.get((code, circuit), 0)
            damp_effect  = np.random.uniform(-0.05, 0.08) if is_damp and code in ("HAM", "VER", "ALO") else (
                           np.random.uniform(-0.08, 0.02) if is_damp else 0)

            score = (
                driver_quali * 0.45 +
                team_pace    * 0.40 +
                affinity     * 0.10 +
                damp_effect  * 0.05 +
                np.random.normal(0, 0.025)
            )
            scores[code] = score

        sorted_drivers = sorted(scores, key=lambda c: scores[c], reverse=True)
        positions = {code: pos + 1 for pos, code in enumerate(sorted_drivers)}

        for code, info in DRIVERS.items():
            rows.append({
                "session_id":    session_id,
                "circuit":       circuit,
                "driver":        code,
                "team":          info["team"],
                "quali_skill":   info["quali_skill"],
                "team_pace":     TEAM_QUALI_PACE[info["team"]],
                "traction":      traits["traction"],
                "braking":       traits["braking"],
                "aero":          traits["aero"],
                "track_evo":     traits["track_evo"],
                "temperature":   round(temp, 1),
                "humidity":      round(humidity, 1),
                "is_damp":       is_damp,
                "affinity":      DRIVER_CIRCUIT_AFFINITY.get((code, circuit), 0),
                "grid_position": positions[code],
                "pole":          1 if positions[code] == 1 else 0,
                "top3":          1 if positions[code] <= 3 else 0,
            })

    return pd.DataFrame(rows)
