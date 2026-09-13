"""Pause & Protect — a K PLUS concept demo.

Shows one persona's story: a real-time risk warning at the moment of transfer
confirmation, instead of only after-the-fact fraud reporting.
"""

import streamlit as st

from llm_explainer import generate_warning
from model import (
    build_features,
    classify_scam_pattern,
    compute_risk_scorecard,
    is_cold_start,
    predict_risk,
)
from train_model import PERSONAS

st.set_page_config(page_title="Pause & Protect", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    /* Referencing K PLUS's own dark home-screen: dark slate app body, white
       "island" cards for key info (balance, intro), teal accent bars on
       section headers, circular avatar + wordmark topbar, bottom nav. */
    .block-container {max-width: 460px; padding-top: 1.5rem;}
    .phone-card {
        background: #1E2A2E; border-radius: 24px; padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35); border: 1px solid #33454B;
    }
    .status-bar {
        display: flex; justify-content: space-between; font-size: 0.75rem;
        color: #9CA3AF; padding: 0 0.1rem 0.6rem 0.1rem; font-weight: 600;
    }
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 1.1rem;
    }
    .avatar {
        width: 38px; height: 38px; border-radius: 50%; background: #33454B;
        display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
        border: 2px solid #34D399; flex-shrink: 0;
    }
    .brand-mark { font-size: 1.3rem; font-weight: 800; color: #F3F4F6; letter-spacing: 0.01em; }
    .brand-mark .plus { color: #34D399; }
    .topbar-icons { font-size: 1rem; color: #9CA3AF; letter-spacing: 0.55rem; }

    .section-header {
        display: flex; align-items: center; justify-content: space-between;
        margin: 1.1rem 0 0.6rem 0;
    }
    .section-header .title {
        display: flex; align-items: center; gap: 0.5rem;
        font-weight: 700; font-size: 0.92rem; color: #F3F4F6;
    }
    .section-header .bar {
        width: 4px; height: 15px; background: #34D399; border-radius: 2px; display: inline-block;
    }
    .section-header .right { font-size: 0.76rem; color: #6EE7B7; }

    .intro-card, .balance-card {
        background: #F9FAFB; border-radius: 14px; padding: 1rem 1.1rem; color: #1F2937;
    }
    .intro-card { display: flex; gap: 0.8rem; align-items: center; }
    .intro-icon {
        width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
        background: linear-gradient(135deg, #0C8B44, #34D399); color: white;
        display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
    }
    .intro-text b { display: block; margin-bottom: 0.15rem; }
    .intro-text span { font-size: 0.8rem; color: #4B5563; }

    .balance-label {
        font-size: 0.72rem; color: #6B7280; text-transform: uppercase;
        letter-spacing: 0.04em; font-weight: 700;
    }
    .balance-amount { font-size: 1.7rem; font-weight: 700; color: #111827; margin-top: 0.1rem; }

    .warning-card, .success-card, .cancel-card, .confirm-card, .danger-card {
        border-radius: 14px; padding: 1rem 1.25rem; margin: 1rem 0;
        color: #1F2937; /* these stay light "island" cards on the dark page, like the reference's white cards */
    }
    .warning-card { background: #FFF7ED; border: 1px solid #FDBA74; }
    .success-card { background: #ECFDF5; border: 1px solid #6EE7B7; }
    .cancel-card { background: #F3F4F6; border: 1px solid #D1D5DB; }
    .confirm-card { background: #EFF6FF; border: 1px solid #BFDBFE; }
    .danger-card { background: #FEF2F2; border: 1px solid #FCA5A5; }

    .gauge-track {
        background: #3A4C52; border-radius: 999px; height: 10px; overflow: hidden; margin-top: 0.4rem;
    }
    .gauge-fill {
        height: 100%; border-radius: 999px;
        transition: width 0.4s ease, background-color 0.4s ease;
    }
    .gauge-label { font-size: 0.78rem; color: #9CA3AF; margin-top: 0.3rem; }
    .pattern-tag {
        display: inline-block; background: #FEF3C7; color: #92400E; font-size: 0.75rem;
        font-weight: 600; padding: 0.3rem 0.65rem; border-radius: 999px; margin: 0.6rem 0 0.2rem 0;
    }
    .activity-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.85rem;
        color: #E5E7EB;
    }
    .activity-amount-out { color: #E5E7EB; font-weight: 600; }
    .activity-amount-in { color: #6EE7B7; font-weight: 600; }

    .bottom-nav {
        display: flex; justify-content: space-around; align-items: flex-end;
        margin: 1.5rem -1.5rem -1.5rem -1.5rem; padding: 0.7rem 0.5rem 0.55rem 0.5rem;
        border-top: 1px solid #33454B;
    }
    .bottom-nav .nav-item { display: flex; flex-direction: column; align-items: center; gap: 0.15rem; font-size: 0.62rem; color: #9CA3AF; }
    .bottom-nav .nav-item.active { color: #34D399; }
    .bottom-nav .nav-icon { font-size: 1.05rem; }
    .bottom-nav .fab {
        width: 44px; height: 44px; border-radius: 50%; background: #34D399; color: #0F1B1E;
        display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
        margin-top: -20px; border: 4px solid #1E2A2E; box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .warning-card, .success-card, .cancel-card, .confirm-card, .danger-card { animation: fadeSlideIn 0.35s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title: str, icon: str = "", right: str = "") -> str:
    label = f"{icon} {title}" if icon else title
    right_html = f'<span class="right">{right}</span>' if right else ""
    return (
        '<div class="section-header"><span class="title">'
        f'<span class="bar"></span>{label}</span>{right_html}</div>'
    )

if "stage" not in st.session_state:
    st.session_state.stage = "idle"
    st.session_state.active_persona_key = "aungnyilatt"
    st.session_state.balances = {key: persona["balance"] for key, persona in PERSONAS.items()}
    st.session_state.transfer = None
    st.session_state.risk_level = None
    st.session_state.proba = None
    st.session_state.factors = []
    st.session_state.message = ""
    st.session_state.pattern_label = ""
    st.session_state.pattern_desc = ""
    st.session_state.scorecard = None

RECENT_ACTIVITY = {
    "aungnyilatt": [
        {"icon": "⚡", "label": "Electric bill", "when": "2 days ago", "amount": -450},
        {"icon": "🏠", "label": "Roommate — rent split", "when": "4 days ago", "amount": -1600},
        {"icon": "👩", "label": "Mom", "when": "6 days ago", "amount": -2000},
        {"icon": "💰", "label": "Salary deposit", "when": "2 weeks ago", "amount": 18000},
    ],
    "add": [
        {"icon": "🍜", "label": "Roommate — split lunch", "when": "3 days ago", "amount": -150},
        {"icon": "💰", "label": "First salary deposit", "when": "2 weeks ago", "amount": 14000},
    ],
}

if "activity" not in st.session_state:
    # A per-session, mutable copy — RECENT_ACTIVITY itself stays the seed data,
    # shared module state that every session would otherwise clobber.
    st.session_state.activity = {key: list(items) for key, items in RECENT_ACTIVITY.items()}

PRESET_SCENARIOS = {
    "aungnyilatt": [
        {
            "label": "🏠 Pay rent to landlord — routine",
            "transfer": {"recipient_id": "condo_landlord", "amount": 3200, "hour": 19, "velocity_24h": 1},
        },
        {
            "label": "🌙 Late-night transfer to a new account",
            "transfer": {"recipient_id": "unknown_night_transfer", "amount": 6400, "hour": 2, "velocity_24h": 2},
        },
        {
            "label": "📈 ฿15,000 to a 'guaranteed returns' contact",
            "transfer": {"recipient_id": "scam_investment_01", "amount": 15000, "hour": 21, "velocity_24h": 4},
        },
    ],
    "add": [
        {
            "label": "🍜 Split a bill with his roommate — routine",
            "transfer": {"recipient_id": "roommate_kbank", "amount": 750, "hour": 13, "velocity_24h": 1},
        },
        {
            "label": "🌙 Late-night transfer to a new account",
            "transfer": {"recipient_id": "unknown_night_transfer", "amount": 1520, "hour": 2, "velocity_24h": 2},
        },
        {
            "label": "📈 ฿3,500 to a 'guaranteed returns' contact",
            "transfer": {"recipient_id": "scam_investment_01", "amount": 3500, "hour": 21, "velocity_24h": 4},
        },
    ],
}


GRADE_COLORS = {"A+": "#10B981", "B+": "#34D399", "C+": "#F59E0B", "D": "#FB923C", "F": "#EF4444"}


def render_grade_badge(scorecard: dict) -> str:
    color = GRADE_COLORS.get(scorecard["grade"], "#6B7280")
    return (
        f'<div style="display:flex; align-items:center; gap:0.5rem; margin:0.5rem 0;">'
        f'<span style="background:{color}; color:white; font-weight:700; font-size:0.95rem; '
        f'padding:0.2rem 0.7rem; border-radius:8px;">{scorecard["grade"]}</span>'
        f'<span style="font-weight:600; color:#374151;">{scorecard["grade_label"]} '
    )


def render_scorecard_breakdown(scorecard: dict) -> str:
    color = GRADE_COLORS.get(scorecard["grade"], "#6B7280")
    rows = ""
    for label, pts, max_pts in scorecard["breakdown"]:
        pct = (pts / max_pts * 100) if max_pts else 0
        rows += (
            '<div style="margin:0.5rem 0;">'
            '<div style="display:flex; justify-content:space-between; font-size:0.82rem;">'
            f'<span>{label}</span><span style="color:#9CA3AF;">{pts}/{max_pts} pts</span></div>'
            f'<div class="gauge-track" style="height:6px;">'
            f'<div class="gauge-fill" style="width:{pct:.0f}%; background:{color};"></div></div>'
            "</div>"
        )
    return rows


def render_gauge(proba: float) -> str:
    pct = max(0.0, min(1.0, proba)) * 100
    if proba < 0.3:
        color = "#10B981"
    elif proba < 0.7:
        color = "#F59E0B"
    else:
        color = "#EF4444"
    return (
        '<div class="gauge-track">'
        f'<div class="gauge-fill" style="width: {pct:.0f}%; background: {color};"></div>'
        "</div>"
        f'<div class="gauge-label">{pct:.0f}% risk</div>'
    )


def start_transfer(transfer: dict) -> None:
    key = st.session_state.active_persona_key
    persona = PERSONAS[key]

    # Insufficient funds is a basic account constraint, not a fraud judgment —
    # check it before the risk model even runs. Also correctly covers a zero
    # balance, since any positive amount exceeds it.
    if transfer["amount"] > st.session_state.balances[key]:
        st.session_state.transfer = transfer
        st.session_state.stage = "insufficient_funds"
        return

    features = build_features(persona, transfer)
    risk_level, proba, factors = predict_risk(features)
    pattern_label, pattern_desc = classify_scam_pattern(transfer["recipient_id"], features)
    scorecard = compute_risk_scorecard(features)

    st.session_state.transfer = transfer
    st.session_state.risk_level = risk_level
    st.session_state.proba = proba
    st.session_state.factors = factors
    st.session_state.pattern_label = pattern_label
    st.session_state.pattern_desc = pattern_desc
    st.session_state.scorecard = scorecard

    if risk_level == "low":
        # Even a known recipient / normal amount still gets a plain confirm
        # step — typos happen regardless of who the money is going to.
        st.session_state.stage = "confirm"
        st.session_state.message = ""
    else:
        st.session_state.stage = "warning"
        st.session_state.message = generate_warning(
            persona["name"], transfer, risk_level, factors, pattern_label
        )


def reset() -> None:
    st.session_state.stage = "idle"
    st.session_state.transfer = None


def switch_persona(key: str) -> None:
    st.session_state.active_persona_key = key
    reset()


def recipient_display_name(persona: dict, recipient_id: str) -> str:
    return persona["known_recipients"].get(recipient_id, recipient_id)


def complete_transfer(next_stage: str) -> None:
    """Actually debit the balance and log it to recent activity — reaching
    this point means the money left the account, whether that was a clean
    send or a 'proceed anyway'."""
    key = st.session_state.active_persona_key
    persona = PERSONAS[key]
    transfer = st.session_state.transfer

    st.session_state.balances[key] -= transfer["amount"]

    recipient_name = recipient_display_name(persona, transfer["recipient_id"])
    icon = "⚠️" if next_stage == "proceeded" else "💸"
    st.session_state.activity[key].insert(0, {
        "icon": icon,
        "label": recipient_name,
        "when": "Just now",
        "amount": -transfer["amount"],
    })

    st.session_state.stage = next_stage


active_persona = PERSONAS[st.session_state.active_persona_key]

st.markdown('<div class="phone-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="topbar">'
    f'<div class="avatar">{"👨" if active_persona["name"] == "Aung" else "🌱"}</div>'
    '<div class="brand-mark">K<span class="plus">+</span></div>'
    '<div class="topbar-icons">🔔&nbsp;&#9211;</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(section_header("Pause & Protect", icon="🛡️"), unsafe_allow_html=True)
st.markdown(
    '<div class="intro-card"><div class="intro-icon">🛡️</div>'
    '<div class="intro-text"><b>A real-time scam warning</b>'
    "<span>Checks every transfer for risk before the money leaves the account.</span></div></div>",
    unsafe_allow_html=True,
)

st.markdown(section_header("Quick Balance", right="Settings &gt;"), unsafe_allow_html=True)
current_balance = st.session_state.balances[st.session_state.active_persona_key]
st.markdown(
    '<div class="balance-card"><div class="balance-label">Available Balance</div>'
    f'<div class="balance-amount">฿{current_balance:,.2f}</div></div>',
    unsafe_allow_html=True,
)
st.space("xxsmall")
persona_cols = st.columns(2)
with persona_cols[0]:
    if st.button(
        "👨‍💼 Aung — 3 months in", use_container_width=True,
        disabled=st.session_state.active_persona_key == "aungnyilatt",
    ):
        switch_persona("aungnyilatt")
        st.rerun()
with persona_cols[1]:
    if st.button(
        "🌱 Add — 2 weeks in", use_container_width=True,
        disabled=st.session_state.active_persona_key == "add",
    ):
        switch_persona("add")
        st.rerun()

st.markdown(section_header(f"Meet {active_persona['name']}", icon="👋"), unsafe_allow_html=True)
st.write(
    f"{active_persona['role']}. Usually transfers around "
    f"฿{active_persona['avg_amount']:,.0f} at a time, mostly to "
    f"{len(active_persona['known_recipients'])} regular people and bills."
)

if is_cold_start(active_persona):
    st.caption(
        f"🌱 {active_persona['name']} is too new to K PLUS to trust his own history alone — "
        "risk is judged against typical first-jobber activity until he builds up more of it, "
        "so being new doesn't get him flagged by itself."
    )

with st.expander(f"📜 {active_persona['name']}'s recent activity"):
    for item in st.session_state.activity[st.session_state.active_persona_key]:
        amount_class = "activity-amount-in" if item["amount"] > 0 else "activity-amount-out"
        amount_text = f"+฿{item['amount']:,.0f}" if item["amount"] > 0 else f"-฿{abs(item['amount']):,.0f}"
        st.markdown(
            f'<div class="activity-row"><span>{item["icon"]} {item["label"]} · '
            f'<span style="color:#9CA3AF;">{item["when"]}</span></span>'
            f'<span class="{amount_class}">{amount_text}</span></div>',
            unsafe_allow_html=True,
        )

if st.session_state.stage == "idle":
    st.markdown(section_header("Try a Transfer", icon="💸"), unsafe_allow_html=True)
    for scenario in PRESET_SCENARIOS[st.session_state.active_persona_key]:
        if st.button(scenario["label"], use_container_width=True):
            start_transfer(scenario["transfer"])
            st.rerun()

    with st.expander("🔍 Try your own transfer — watch the risk update live"):
        saved_contacts = list(active_persona["known_recipients"].items())
        contact_labels = [name for _, name in saved_contacts] + ["➕ Someone new"]
        contact_choice = st.selectbox("Send to", contact_labels)

        if contact_choice == "➕ Someone new":
            recipient_id = st.text_input(
                "Phone number or PromptPay ID", value="089-123-4567",
                help="Anyone not already in the contact list above counts as a first-time recipient.",
            )
        else:
            recipient_id = next(rid for rid, name in saved_contacts if name == contact_choice)

        amount = st.number_input("Amount (THB)", min_value=1.0, value=1000.0, step=100.0)
        hour = st.slider("Hour of day", 0, 23, 14)
        velocity = st.number_input(
            "Transfers in the last 24h (including this one)", min_value=1, max_value=10, value=1
        )

        live_transfer = {
            "recipient_id": recipient_id, "amount": amount, "hour": hour, "velocity_24h": velocity,
        }
        live_features = build_features(active_persona, live_transfer)
        _, live_proba, live_factors = predict_risk(live_features)
        live_scorecard = compute_risk_scorecard(live_features)

        st.markdown(render_gauge(live_proba), unsafe_allow_html=True)
        st.markdown(render_grade_badge(live_scorecard), unsafe_allow_html=True)
        over_balance = live_transfer["amount"] > current_balance
        if over_balance:
            st.caption(f"🚫 Not enough balance — only ฿{current_balance:,.2f} available.")
        else:
            st.caption(live_factors[0] if live_factors else "No red flags detected yet.")

        if st.button(
            "Send Transfer", type="primary", use_container_width=True, disabled=over_balance,
        ):
            start_transfer(live_transfer)
            st.rerun()

elif st.session_state.stage == "insufficient_funds":
    t = st.session_state.transfer
    balance_now = st.session_state.balances[st.session_state.active_persona_key]
    shortfall = t["amount"] - balance_now
    st.markdown(
        '<div class="cancel-card">🚫 <b>Not enough balance</b><br><br>'
        f'This transfer needs ฿{t["amount"]:,.0f}, but only ฿{balance_now:,.2f} is available '
        f'(short by ฿{shortfall:,.2f}). No risk check runs — this is rejected before it gets that far, '
        "the same way a real bank would.</div>",
        unsafe_allow_html=True,
    )
    st.button("Back", on_click=reset, use_container_width=True)

elif st.session_state.stage == "confirm":
    t = st.session_state.transfer
    st.markdown(render_gauge(st.session_state.proba), unsafe_allow_html=True)
    st.markdown(render_grade_badge(st.session_state.scorecard), unsafe_allow_html=True)
    st.markdown(
        '<div class="confirm-card">✋ <b>Confirm this transfer</b><br><br>'
        f'Send <b>฿{t["amount"]:,.0f}</b> to <b>{recipient_display_name(active_persona, t["recipient_id"])}</b>?<br>'
        '<span style="color:#6B7280; font-size:0.85rem;">No red flags found — but double-check the '
        "amount before sending. Even known accounts are worth a second look.</span></div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.stage = "cancelled"
            st.rerun()
    with col2:
        if st.button("Confirm & Send", type="primary", use_container_width=True):
            complete_transfer("sent")
            st.rerun()

elif st.session_state.stage == "sent":
    t = st.session_state.transfer
    st.markdown(render_gauge(st.session_state.proba), unsafe_allow_html=True)
    st.markdown(render_grade_badge(st.session_state.scorecard), unsafe_allow_html=True)
    st.markdown(
        f'<div class="success-card">✅ <b>Sent ฿{t["amount"]:,.0f}</b> to {recipient_display_name(active_persona, t["recipient_id"])}. '
        "No red flags — straight through, like today's K PLUS.</div>",
        unsafe_allow_html=True,
    )
    st.button("Try another scenario", on_click=reset, use_container_width=True)

elif st.session_state.stage == "warning":
    st.markdown(render_gauge(st.session_state.proba), unsafe_allow_html=True)
    st.markdown(render_grade_badge(st.session_state.scorecard), unsafe_allow_html=True)
    st.markdown(
        f'<span class="pattern-tag">🔎 {st.session_state.pattern_label}</span>',
        unsafe_allow_html=True,
    )
    is_high = st.session_state.risk_level == "high"
    card_class = "danger-card" if is_high else "warning-card"
    icon = "🚨" if is_high else "⚠️"
    st.markdown(
        f'<div class="{card_class}">{icon} <b>{st.session_state.risk_level.title()} risk</b>'
        f'<br><br>{st.session_state.message}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why is this flagged?"):
        st.write(st.session_state.pattern_desc)
        st.caption("Risk scorecard — a transparent point breakdown, weighted from what the model itself learned matters most:")
        st.markdown(render_scorecard_breakdown(st.session_state.scorecard), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel transfer", use_container_width=True):
            st.session_state.stage = "cancelled"
            st.rerun()
    with col2:
        if st.button("Proceed anyway", use_container_width=True):
            complete_transfer("proceeded")
            st.rerun()

elif st.session_state.stage == "cancelled":
    st.markdown(
        '<div class="cancel-card">🙌 Transfer cancelled. Good call — take your time to double check.</div>',
        unsafe_allow_html=True,
    )
    st.button("Try another scenario", on_click=reset, use_container_width=True)

elif st.session_state.stage == "proceeded":
    t = st.session_state.transfer
    is_high = st.session_state.risk_level == "high"
    card_class = "danger-card" if is_high else "warning-card"
    icon = "🚨" if is_high else "⚠️"
    st.markdown(
        f'<div class="{card_class}">{icon} Sent ฿{t["amount"]:,.0f} to {recipient_display_name(active_persona, t["recipient_id"])} anyway. '
        "(In the real product, this creates a flagged record KBTG can use for faster fraud response.)</div>",
        unsafe_allow_html=True,
    )
    st.button("Try another scenario", on_click=reset, use_container_width=True)

st.markdown(
    '<div class="bottom-nav">'
    '<div class="nav-item active"><span class="nav-icon">🏠</span>Home</div>'
    '<div class="nav-item"><span class="nav-icon">🛒</span>Market</div>'
    '<div class="fab">🛡️</div>'
    '<div class="nav-item"><span class="nav-icon">▦</span>Scan</div>'
    '<div class="nav-item"><span class="nav-icon">⋯</span>More</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
