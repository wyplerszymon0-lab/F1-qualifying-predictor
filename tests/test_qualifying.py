import pytest
import pandas as pd
from data import generate_qualifying_data, DRIVERS, CIRCUIT_QUALI_TRAITS
from model import train_model, predict_qualifying, prepare_data, feature_importance, FEATURES


@pytest.fixture(scope="module")
def dataset():
    return generate_qualifying_data(n_sessions=100, seed=42)


@pytest.fixture(scope="module")
def trained_model(dataset):
    model, acc = train_model(dataset)
    return model, acc


def test_dataset_has_required_columns(dataset):
    required = ["session_id", "driver", "team", "grid_position", "pole", "top3", "circuit"]
    for col in required:
        assert col in dataset.columns


def test_each_session_has_one_pole(dataset):
    poles = dataset.groupby("session_id")["pole"].sum()
    assert (poles == 1).all()


def test_all_drivers_present(dataset):
    for code in DRIVERS:
        assert code in dataset["driver"].values


def test_grid_positions_unique_per_session(dataset):
    for _, group in dataset.groupby("session_id"):
        positions = group["grid_position"].tolist()
        assert len(positions) == len(set(positions))


def test_prepare_data_correct_features(dataset):
    X, y = prepare_data(dataset)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)


def test_model_trains_successfully(trained_model):
    model, acc = trained_model
    assert model is not None
    assert 0.0 < acc <= 1.0


def test_predict_qualifying_returns_all_drivers(trained_model):
    model, _ = trained_model
    result   = predict_qualifying(model, "Britain")
    assert len(result) == len(DRIVERS)


def test_probabilities_sum_to_one(trained_model):
    model, _ = trained_model
    result   = predict_qualifying(model, "Britain")
    total    = result["pole_probability"].sum()
    assert abs(total - 1.0) < 0.001


def test_result_sorted_by_probability(trained_model):
    model, _ = trained_model
    result   = predict_qualifying(model, "Britain")
    probs    = result["pole_probability"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_damp_differs_from_dry(trained_model):
    model, _ = trained_model
    dry  = predict_qualifying(model, "Britain", is_damp=0)
    damp = predict_qualifying(model, "Britain", is_damp=1)
    assert not dry["pole_probability"].equals(damp["pole_probability"])


def test_temperature_affects_prediction(trained_model):
    model, _ = trained_model
    hot  = predict_qualifying(model, "Spain", temperature=40.0)
    cold = predict_qualifying(model, "Spain", temperature=15.0)
    assert not hot["pole_probability"].equals(cold["pole_probability"])


def test_unknown_circuit_raises(trained_model):
    model, _ = trained_model
    with pytest.raises(ValueError, match="Unknown circuit"):
        predict_qualifying(model, "Atlantis GP")


def test_feature_importance_complete(trained_model):
    model, _ = trained_model
    fi = feature_importance(model)
    assert set(fi["feature"]) == set(FEATURES)
    assert abs(fi["importance"].sum() - 1.0) < 0.001


def test_monaco_affinity_boosts_leclerc(trained_model):
    model, _ = trained_model
    result   = predict_qualifying(model, "Monaco")
    lec_rank = result[result["driver"] == "LEC"].index[0]
    assert lec_rank <= 3
```

---

**`requirements.txt`**
```
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
pytest==8.2.0
