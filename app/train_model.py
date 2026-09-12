"""Synthetic data generation and risk-model training for Pause & Protect."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"

FEATURE_COLUMNS = [
    "is_first_time_recipient",
    "amount_ratio",
    "hour",
    "is_night",
    "velocity_24h",
    "recipient_flagged_count",
]

# Stand-in for KBTG's own reported-scam-account history (fed by the existing
# in-app fraud-report channel in the real product).
FLAGGED_RECIPIENTS = {
    "scam_investment_01": 5,
    "scam_investment_02": 3,
    "scam_jobscam_01": 4,
    "scam_qrclone_01": 2,
}

# The established persona shown in the live demo.
# known_recipients maps recipient_id -> a display name for the recipient picker;
# elsewhere in the code it's used the same way a set would be (`in`, `len()`).
DEFAULT_PERSONA = {
    "name": "Aung",
    "age": 25,
    "role": "Marketing assistant, 3 months into his first job",
    "avg_amount": 3200.0,
    "transaction_count": 45,  # plenty of personal history — his own baseline is trusted outright
    "balance": 24850.0,
    "known_recipients": {
        "mom_kbank": "Mom",
        "roommate_split": "Roommate (rent split)",
        "condo_landlord": "Condo Landlord",
        "electric_bill": "Electric Company",
    },
}

# A brand-new persona used to demonstrate the cold-start problem: a first-jobber
# who has barely used K PLUS yet has almost no personal history to compare against.
NEW_USER_PERSONA = {
    "name": "Add",
    "age": 22,
    "role": "Just started his first job, 2 weeks into using K PLUS",
    "avg_amount": 200.0,  # thin and noisy — only a few small transactions so far
    "transaction_count": 3,
    "balance": 9200.0,
    "known_recipients": {
        "roommate_kbank": "Roommate",
    },
}

PERSONAS = {"aungnyilatt": DEFAULT_PERSONA, "add": NEW_USER_PERSONA}

# How many of a user's own transactions we need before trusting their personal
# average outright. Below this, the model blends in a peer-cohort baseline so a
# brand-new user isn't flagged simply for lacking history yet.
COLD_START_THRESHOLD = 20

# Typical transfer size for a first-jobber who is new to K PLUS — used to fill
# in for the personal baseline a new user hasn't built up yet.
PEER_COHORT_AVG_AMOUNT = 850.0


def _sample_normal_transaction() -> dict:
    hour = int(np.clip(RNG.normal(14, 6), 0, 23))
    return {
        "is_first_time_recipient": int(RNG.random() < 0.2),
        # Lognormal, not normal+clip: clipping a normal distribution piles up
        # a point mass exactly at the clip boundary, which a tree model will
        # latch onto as a spuriously perfect signal. Lognormal stays positive
        # and has no artificial boundary.
        "amount_ratio": float(RNG.lognormal(mean=0.0, sigma=0.5)),
        "hour": hour,
        "is_night": int(hour < 6 or hour >= 23),
        "velocity_24h": int(RNG.poisson(1.2)),
        "recipient_flagged_count": 0,
        "is_scam": 0,
    }


def _sample_risky_transaction() -> dict:
    hour = int(np.clip(RNG.normal(2, 5), 0, 23)) if RNG.random() < 0.6 else int(RNG.integers(0, 24))
    return {
        "is_first_time_recipient": int(RNG.random() < 0.75),
        "amount_ratio": float(RNG.lognormal(mean=1.386, sigma=0.6)),  # median ratio ~4x
        "hour": hour,
        "is_night": int(hour < 6 or hour >= 23),
        "velocity_24h": int(RNG.poisson(2.5)),
        "recipient_flagged_count": int(
            RNG.choice([0, 1, 2, 3, 4, 5], p=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05])
        ),
        "is_scam": 1,
    }


def generate_dataset(n_normal: int = 3500, n_risky: int = 1500, label_noise: float = 0.04) -> pd.DataFrame:
    rows = [_sample_normal_transaction() for _ in range(n_normal)]
    rows += [_sample_risky_transaction() for _ in range(n_risky)]
    df = pd.DataFrame(rows)

    # Real fraud labels are never perfectly clean (e.g. reported-late or
    # mislabeled cases) — a touch of label noise keeps the demo honest and
    # avoids a suspicious 100%-accuracy model.
    flip_mask = RNG.random(len(df)) < label_noise
    df.loc[flip_mask, "is_scam"] = 1 - df.loc[flip_mask, "is_scam"]

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_and_save() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = generate_dataset()
    df.to_csv(DATA_DIR / "synthetic_transfers.csv", index=False)

    X = df[FEATURE_COLUMNS]
    y = df["is_scam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["legit", "scam"]))

    joblib.dump(model, MODEL_DIR / "risk_model.pkl")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    print(f"Model saved to {MODEL_DIR / 'risk_model.pkl'}")


if __name__ == "__main__":
    train_and_save()
