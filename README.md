# f1-qualifying-predictor

Machine learning model that predicts Formula 1 qualifying results
and pole position probability using Random Forest.

## Features

- RandomForestClassifier trained on simulated qualifying data
- Considers driver qualifying skill, team pace, circuit traits
- Weather conditions — temperature, humidity, damp track
- Circuit-specific driver affinity (e.g. Leclerc at Monaco)
- Pole position probability for all 10 drivers

## Run
```bash
pip install -r requirements.txt

python main.py Britain
python main.py Monaco --temp 22 --humidity 60
python main.py Singapore --damp
python main.py --circuits
python main.py Britain --importance
```

## Example Output
```
Training qualifying model...
Model accuracy: 89.1%

============================================================
  F1 QUALIFYING PREDICTOR — MONACO (DRY)
  Temp: 22.0°C | Humidity: 60.0%
============================================================
Pos  Driver                Team            Pole %
------------------------------------------------------------
🥇   Charles Leclerc       Ferrari         28.4%
🥈   Max Verstappen        Red Bull        24.1%
🥉   Lewis Hamilton        Mercedes        15.3%

  Favourite for pole: Charles Leclerc (28.4%)
```

## Test
```bash
pytest tests/ -v
```

## Project Structure
```
f1-qualifying-predictor/
├── data.py           # Data generation, driver/circuit data
├── model.py          # RandomForest training and prediction
├── main.py           # CLI entry point
├── requirements.txt
├── README.md
└── tests/
    └── test_qualifying.py
```

## Author

**Szymon Wypler**
