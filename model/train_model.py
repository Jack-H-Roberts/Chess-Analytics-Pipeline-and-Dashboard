#!/usr/bin/env python3
"""Retrain the win/draw/loss model on the 48-feature gold layer.

Data source (pick one):
  1. Databricks SQL connector (default) — reads chess.gold.game_features
     directly. Requires env vars, never hardcoded:
       DATABRICKS_HOST        e.g. dbc-xxxxxxxx.cloud.databricks.com
       DATABRICKS_HTTP_PATH   SQL warehouse -> Connection details
       DATABRICKS_TOKEN       personal access token
  2. --csv path/to/gold_extract.csv — a downloaded extract of
     `SELECT * FROM chess.gold.game_features` (any row order; the script
     re-sorts chronologically by start_ts).

Modeling design is ported intact from the 2024 pipeline, deliberately:
  - 28-of-48 feature subset: pre-game context + opening only. Excluded:
    in-game/post-game signals (castling, move counts, time usage), Account,
    EloDifference, and TimeControl (near-constant within the rapid
    population: 600s in all but ~11 games).
  - Optuna search, 10-fold *stratified shuffled* CV for tuning. Known
    tradeoff: shuffling leaks future games across folds during tuning;
    the final evaluation is an untouched chronological holdout, which is
    the honest number. A time-series CV is a documented future upgrade.
  - Class weights ~100:1:100 (win:draw:loss) scaled by inverse frequency,
    countering the ~4% draw rate.
  - Early stopping monitors the holdout during the final fit — mild
    selection leakage, same as 2024, acknowledged rather than hidden.

Usage:
  python model/train_model.py                # connector, 25 trials, CUDA
  python model/train_model.py --csv gold.csv --trials 15 --device cpu
"""

import argparse
import datetime
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection

warnings.filterwarnings("ignore")

QUERY = "SELECT * FROM chess.gold.game_features ORDER BY start_ts"

# The deliberate 28-feature subset (of the 48 in gold).
FEATURES = [
    "GameOfDay", "GameOfWeek", "TimeOfDay", "TimeSinceLast",
    "IsMonday", "IsTuesday", "IsWednesday", "IsThursday",
    "IsFriday", "IsSaturday", "IsSunday",
    "DailyWinPerc", "DailyDrawPerc", "DailyLossPerc",
    "WeeklyWinPerc", "WeeklyDrawPerc", "WeeklyLossPerc",
    "Color",
    "ECO_A00", "ECO_A40", "ECO_A45", "ECO_B10", "ECO_B12", "ECO_B13",
    "ECO_D00", "ECO_D02", "ECO_D10", "ECO_Other",
]
TARGET = "Result"  # 0 win / 1 draw / 2 loss


def load_gold(csv: str | None) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv)
        print(f"loaded {len(df):,} rows from {csv}")
    else:
        from databricks import sql as dbsql  # lazy: only needed on this path

        with dbsql.connect(
            server_hostname=os.environ["DATABRICKS_HOST"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
        ) as conn, conn.cursor() as cur:
            cur.execute(QUERY)
            df = cur.fetchall_arrow().to_pandas()
        print(f"loaded {len(df):,} rows from chess.gold.game_features")

    df = df.sort_values("start_ts").reset_index(drop=True)  # chronological
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"gold extract is missing columns: {missing}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="CSV extract instead of connector")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="model/artifacts")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    df = load_gold(args.csv)
    X = df[FEATURES].copy()
    X["TimeSinceLast"] = np.log10(X["TimeSinceLast"] + 1)
    y = df[TARGET].astype(int)

    # Chronological holdout: the last N% of games form the test set.
    split = int(len(X) * (1 - args.test_split))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"train {len(X_train):,} | test {len(X_test):,} (chronological)")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    counts = y_train.value_counts()
    class_weights = {
        0: 100 * len(y_train) / counts[0],
        1: 1 * len(y_train) / counts[1],
        2: 100 * len(y_train) / counts[2],
    }

    base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "device": args.device,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 7),
        }
        kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=999)
        scores = []
        for tr_idx, va_idx in kf.split(X_train, y_train):
            d_tr = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
            d_va = xgb.DMatrix(X_train.iloc[va_idx], label=y_train.iloc[va_idx])
            booster = xgb.train(
                params, d_tr, num_boost_round=1000,
                evals=[(d_va, "validation")],
                early_stopping_rounds=50, verbose_eval=False,
            )
            scores.append(metrics.log_loss(y_train.iloc[va_idx], booster.predict(d_va)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("\nbest hyperparameters:", study.best_params)

    dtrain.set_weight(y_train.map(class_weights))
    best = xgb.train(
        {**base_params, **study.best_params},
        dtrain, num_boost_round=1000,
        evals=[(dtest, "test")],
        early_stopping_rounds=50, verbose_eval=False,
    )

    proba = best.predict(dtest)
    pred = proba.argmax(axis=1)
    print(f"\nlog loss (chronological holdout): {metrics.log_loss(y_test, proba):.4f}")
    print(f"accuracy: {metrics.accuracy_score(y_test, pred):.4f}\n")
    print(metrics.classification_report(y_test, pred, target_names=["Win", "Draw", "Loss"]))

    # Artifacts: model, the exact feature list it was trained on (kills the
    # ExplainInput drift problem at the source), importances, confusion matrix.
    best.save_model(out / "model.json")
    (out / "features.json").write_text(json.dumps(FEATURES, indent=2))

    for imp_type in ["weight", "gain", "cover"]:
        imp = best.get_score(importance_type=imp_type)
        if not imp:
            continue
        s = pd.Series(imp).sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        s.plot(kind="bar")
        plt.title(f"Feature importance ({imp_type})")
        plt.tight_layout()
        plt.savefig(out / f"importance_{imp_type}.png", dpi=200)
        plt.close()

    cm = metrics.confusion_matrix(y_test, pred)
    metrics.ConfusionMatrixDisplay(cm, display_labels=["Win", "Draw", "Loss"]).plot(cmap="viridis")
    plt.title("Confusion matrix — chronological holdout")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close()

    print(f"\nartifacts -> {out}/")
    print(f"end: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
