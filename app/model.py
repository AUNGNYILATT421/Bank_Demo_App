"""Feature engineering and risk prediction for a single transfer."""

from pathlib import Path

import joblib
import pandas as pd

from train_model import (
    COLD_START_THRESHOLD,
    FEATURE_COLUMNS,
    FLAGGED_RECIPIENTS,
    MODEL_DIR,
    PEER_COHORT_AVG_AMOUNT,
)

MODEL_PATH = MODEL_DIR / "risk_model.pkl"

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def effective_avg_amount(persona: dict) -> float:
    """The baseline a transfer's amount is judged against.

    A user with plenty of personal history is judged against their own
    average outright. A brand-new user's thin, noisy average is blended with
    a peer-cohort baseline instead, weighted by how much personal history
    they actually have — so being new doesn't by itself look risky.
    """
    personal_avg = persona["avg_amount"]
    personal_count = persona.get("transaction_count", COLD_START_THRESHOLD)
    if personal_count >= COLD_START_THRESHOLD:
        return personal_avg
    weight = personal_count / COLD_START_THRESHOLD
    return weight * personal_avg + (1 - weight) * PEER_COHORT_AVG_AMOUNT


def is_cold_start(persona: dict) -> bool:
    return persona.get("transaction_count", COLD_START_THRESHOLD) < COLD_START_THRESHOLD


def build_features(persona: dict, transfer: dict) -> dict:
    recipient_id = transfer["recipient_id"]
    hour = transfer["hour"]
    return {
        "is_first_time_recipient": int(recipient_id not in persona["known_recipients"]),
        "amount_ratio": transfer["amount"] / effective_avg_amount(persona),
        "hour": hour,
        "is_night": int(hour < 6 or hour >= 23),
        "velocity_24h": transfer.get("velocity_24h", 1),
        "recipient_flagged_count": FLAGGED_RECIPIENTS.get(recipient_id, 0),
    }


def _contributing_factors(features: dict) -> list:
    factors = []
    if features["is_first_time_recipient"]:
        factors.append("This is the first time you're sending money to this account.")
    if features["amount_ratio"] >= 2:
        factors.append(
            f"This transfer is about {features['amount_ratio']:.1f}x larger than your usual amount."
        )
    if features["is_night"]:
        factors.append("You're sending this late at night.")
    if features["velocity_24h"] >= 3:
        factors.append("You've made several transfers in the last 24 hours.")
    if features["recipient_flagged_count"] > 0:
        factors.append("Other K PLUS users have reported this account before.")
    return factors


def classify_scam_pattern(recipient_id: str, features: dict) -> tuple:
    """Match a transfer against known Thai scam typologies (BOT mule-account
    reporting + PromptPay scam patterns researched for this pitch), not just
    a bare risk number — so the tag shown to the user is grounded in real
    fraud categories rather than decoration.
    """
    # Require an actual prior report before trusting the substring match on the
    # ID — otherwise a coincidentally-named but unflagged "new recipient" typed
    # in live (e.g. containing "investment") would get mislabeled.
    is_flagged = features["recipient_flagged_count"] > 0
    if is_flagged and "investment" in recipient_id:
        return ("Investment scam pattern", "Matches accounts reported for 'guaranteed returns' investment scams.")
    if is_flagged and "jobscam" in recipient_id:
        return ("Job-offer scam pattern", "Matches accounts reported in fake job-offer / advance-fee scams.")
    if is_flagged and "qrclone" in recipient_id:
        return ("QR-clone / bank-impersonation pattern", "Matches accounts linked to cloned QR codes or fake bank-support requests.")
    if is_flagged:
        return ("Reported-account pattern", "This account has prior reports from other K PLUS users.")
    if features["is_first_time_recipient"] and features["amount_ratio"] >= 3 and features["is_night"]:
        return ("Urgent late-night request pattern", "Large, late-night transfers to new accounts are common in romance and urgent-favor scams.")
    if features["is_first_time_recipient"] and features["velocity_24h"] >= 3:
        return ("Rapid multi-transfer pattern", "Several transfers in a short window to a new recipient resembles mule-account activity.")
    if features["is_first_time_recipient"] and features["is_night"]:
        return ("Late-night new-recipient pattern", "First-time transfers late at night are a common early sign in romance and urgent-favor scams, even at modest amounts.")
    return ("General unusual-activity pattern", "This transfer doesn't match your usual behavior, but no specific scam type is confirmed.")


# --- Transparent, human-readable risk scorecard -----------------------------
# A second, independent explanation of risk, alongside the ML model rather
# than instead of it: point weights per factor, graded like a credit score.
# The weights aren't invented — they're the trained GradientBoostingClassifier's
# own feature_importances_, rounded to a 100-point scale so anyone, not just
# a data scientist, can check the arithmetic by hand:
#   amount_ratio             0.74  -> 74 points
#   recipient_flagged_count  0.16  -> 16 points
#   is_first_time_recipient  0.05  ->  5 points
#   hour / is_night          0.04  ->  4 points (folded into one "late-night" factor
#                                     — is_night alone carried ~0% importance on its
#                                     own, since it's a coarse cut of `hour`)
#   velocity_24h             0.01  ->  1 point
RISK_GRADE_BANDS = [
    (15, "A+", "Very Safe"),
    (35, "B+", "Mostly Safe"),
    (55, "C+", "Some Caution"),
    (75, "D", "Risky"),
    (101, "F", "High Risk"),
]


def _amount_ratio_points(ratio: float) -> int:
    if ratio < 1.5:
        return 0
    if ratio < 2:
        return 15
    if ratio < 3:
        return 35
    if ratio < 5:
        return 55
    return 74


def _flagged_points(count: int) -> int:
    if count >= 5:
        return 16
    if count >= 3:
        return 12
    if count >= 1:
        return 8
    return 0


def compute_risk_scorecard(features: dict) -> dict:
    breakdown = [
        ("Unusual transfer amount", _amount_ratio_points(features["amount_ratio"]), 74),
        ("Reported by other users", _flagged_points(features["recipient_flagged_count"]), 16),
        ("First-time recipient", 5 if features["is_first_time_recipient"] else 0, 5),
        ("Late-night timing", 4 if features["is_night"] else 0, 4),
        ("Several recent transfers", 1 if features["velocity_24h"] >= 3 else 0, 1),
    ]
    score = sum(points for _, points, _ in breakdown)
    grade, grade_label = next(
        (g, label) for cutoff, g, label in RISK_GRADE_BANDS if score < cutoff
    )
    return {"score": score, "grade": grade, "grade_label": grade_label, "breakdown": breakdown}


def predict_risk(features: dict):
    model = _load_model()
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    proba = float(model.predict_proba(row)[0, 1])

    if proba < 0.3:
        risk_level = "low"
    elif proba < 0.7:
        risk_level = "medium"
    else:
        risk_level = "high"

    factors = _contributing_factors(features)
    return risk_level, proba, factors
