"""Turns a risk score into a plain-language warning via Claude, with a safe fallback.

The fallback exists so a live demo never visibly breaks if the API call fails
(no network, bad key, timeout) — see the hackathon build plan for why this matters.
"""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
except ImportError:
    anthropic = None

_client = None
_client_checked = False


def _get_client():
    global _client, _client_checked
    if not _client_checked:
        _client_checked = True
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic is not None and api_key:
            _client = anthropic.Anthropic(api_key=api_key, timeout=8.0)
    return _client


def _fallback_message(
    persona_name: str, risk_level: str, contributing_factors: list, pattern_label: str = ""
) -> str:
    factors_text = " ".join(contributing_factors) if contributing_factors else "Something about this transfer looks unusual."
    pattern_text = f" This matches a **{pattern_label}** we've seen reported before." if pattern_label else ""
    return (
        f"{persona_name}, this transfer looks {risk_level}-risk.{pattern_text} {factors_text} "
        f"Take a moment before confirming — is this someone you know and trust?"
    )


def generate_warning(
    persona_name: str,
    transfer: dict,
    risk_level: str,
    contributing_factors: list,
    pattern_label: str = "",
) -> str:
    client = _get_client()
    if client is None:
        return _fallback_message(persona_name, risk_level, contributing_factors, pattern_label)

    pattern_line = f" It also matches a known '{pattern_label}'." if pattern_label else ""
    prompt = (
        "You are a calm, friendly safety assistant inside a mobile banking app. "
        f"A user named {persona_name} is about to send {transfer['amount']:.0f} THB to a recipient. "
        f"A risk model flagged this transfer as {risk_level} risk because: "
        f"{' '.join(contributing_factors) or 'the pattern looks unusual for this user.'}"
        f"{pattern_line} "
        f"Write a single short warning (max 2 sentences, plain language, no jargon, no scolding tone) "
        f"that helps {persona_name} pause and think before confirming. "
        "Speak like a helpful person, not a system — do not mention 'model', 'algorithm', or 'AI'."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return text or _fallback_message(persona_name, risk_level, contributing_factors, pattern_label)
    except Exception:
        return _fallback_message(persona_name, risk_level, contributing_factors, pattern_label)
