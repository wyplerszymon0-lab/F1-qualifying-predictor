import argparse
import sys
from data import generate_qualifying_data, CIRCUIT_QUALI_TRAITS
from model import train_model, predict_qualifying, feature_importance


def print_qualifying(result, circuit, temp, humidity, is_damp):
    cond = "DAMP" if is_damp else "DRY"
    print(f"\n{'='*60}")
    print(f"  F1 QUALIFYING PREDICTOR — {circuit.upper()} ({cond})")
    print(f"  Temp: {temp}°C | Humidity: {humidity}%")
    print(f"{'='*60}")
    print(f"{'Pos':<5}{'Driver':<22}{'Team':<16}{'Pole %'}")
    print(f"{'-'*60}")

    for i, row in result.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        print(f"{medal:<5}{row['name']:<22}{row['team']:<16}{row['pole_pct']}%")

    print(f"{'='*60}")
    fav = result.iloc[0]
    print(f"\n  Favourite for pole: {fav['name']} ({fav['pole_pct']}%)\n")


def main():
    parser = argparse.ArgumentParser(description="F1 Qualifying Predictor")
    parser.add_argument("circuit",     nargs="?",   default="Britain")
    parser.add_argument("--temp",      type=float,  default=25.0,  help="Track temp (°C)")
    parser.add_argument("--humidity",  type=float,  default=50.0,  help="Humidity (%)")
    parser.add_argument("--damp",      action="store_true",         help="Damp track")
    parser.add_argument("--circuits",  action="store_true",         help="List circuits")
    parser.add_argument("--importance",action="store_true",         help="Feature importance")
    args = parser.parse_args()

    if args.circuits:
        print("\nAvailable circuits:")
        for c in CIRCUIT_QUALI_TRAITS:
            traits = CIRCUIT_QUALI_TRAITS[c]
            print(f"  {c:<15} traction={traits['traction']} aero={traits['aero']}")
        return

    print("Training qualifying model...")
    df       = generate_qualifying_data(n_sessions=300)
    model, acc = train_model(df)
    print(f"Model accuracy: {acc:.1%}\n")

    if args.importance:
        fi = feature_importance(model)
        print("Feature Importance:")
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['feature']:<15} {bar} {row['importance']:.3f}")
        print()

    try:
        result = predict_qualifying(
            model, args.circuit,
            temperature=args.temp,
            humidity=args.humidity,
            is_damp=int(args.damp),
        )
        print_qualifying(result, args.circuit, args.temp, args.humidity, args.damp)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
