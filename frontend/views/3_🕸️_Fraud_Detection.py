"""
3_🕸️_Fraud_Detection.py  —  GNN Fraud Detection Lab

Six tabs:
  1. Test Transaction      — custom scoring + local graph + explanation
  2. Teaching Scenarios    — 3 pre-defined + custom, side-by-side comparison
  3. Graph Explorer        — schema, ring diagram, dataset statistics
  4. Why GNN?              — Traditional ML vs GNN, message-passing walkthrough
  5. How It Works          — Architecture, SAGEConv, Trace One Transaction
  6. Model & Limitations   — Status, parameters, limitations, how to train

⚠ SYNTHETIC DEMONSTRATION — not trained on real customer transaction data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_kit import backend_url  # noqa: E402

BACKEND = backend_url()  # set in the sidebar 🔌 panel

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Fraud Detection Lab — CAMEL Sentinel",
    page_icon="🕸️",
    layout="wide",
)

# ── Persistent warning banner ─────────────────────────────────────────────────
st.error(
    "⚠ **SYNTHETIC DEMONSTRATION** — This module uses artificially generated "
    "transaction data with deliberately planted fraud patterns. It is **not** "
    "trained on real customer transactions and does **not** represent actual "
    "fraud rates or real banking behaviour. For educational purposes only."
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🕸️ GNN Fraud Detection Lab")
st.markdown(
    "This module demonstrates how a **Graph Neural Network** detects fraud by "
    "learning from transaction features **and** the relationships among customers, "
    "accounts, merchants, and shared devices — relationships that a conventional "
    "tabular model cannot see."
)

# ── Backend status ────────────────────────────────────────────────────────────
try:
    _status = requests.get(f"{BACKEND}/gnn/status", timeout=3).json()
    model_ready = _status.get("gnn_model_loaded", False)
except Exception:
    model_ready = False
    _status = {}

if model_ready:
    st.success("GNN model loaded and ready.")
else:
    st.warning(
        "**GNN model not loaded.** "
        "Run `notebooks/04_gnn_fraud_detection.ipynb` end-to-end, then restart "
        "the backend.  Static sections (Graph Explorer, Why GNN?, How It Works) "
        "are available without a loaded model."
    )


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH VISUALISATION HELPER
# Builds a local conceptual graph from input parameters using networkx.
# Labelled as "conceptual" because single-transaction inference uses a
# minimal subgraph — not the full training graph.
# ═════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "customer":    "#4C9BE8",   # blue
    "account":     "#56B87A",   # green
    "transaction": "#E86B4C",   # red-orange  (target)
    "merchant":    "#9B59B6",   # purple
    "device":      "#F5A623",   # amber
    "peer_acct":   "#AACF88",   # light green  (neighbouring accounts)
    "fraud_txn":   "#E84C6B",   # crimson      (known-fraudulent neighbour)
}

EDGE_COLOR = "#888888"
RING_EDGE_COLOR = "#E84C6B"


def _draw_local_graph(
    amount: float,
    avg_amount: float,
    hour: int,
    is_cross_border: bool,
    merchant_risk: float,
    device_accounts: int,
    device_risk: float,
    velocity: float,
    balance: float,
) -> plt.Figure:
    """
    Render a conceptual local graph around a single transaction.

    Nodes: customer, account (target), transaction (target),
           merchant, device, peer accounts (if device is shared).
    All relationships reflect what was ENTERED — not invented.
    Labelled as 'Conceptual visualization based on entered values'.
    """
    G = nx.DiGraph()

    # Core nodes
    G.add_node("Customer",      ntype="customer",    label="Customer\ntenure/age/risk")
    G.add_node("Account",       ntype="account",     label=f"Account\nbal=${balance:,.0f}\nvel={velocity:.1f}/day")
    G.add_node("Transaction",   ntype="transaction", label=f"Transaction\n${amount:,.2f}\nhour={hour}")
    G.add_node("Merchant",      ntype="merchant",    label=f"Merchant\nrisk={merchant_risk:.2f}")
    G.add_node("Device",        ntype="device",      label=f"Device\n{device_accounts} acct(s)\nrisk={device_risk:.2f}")

    # Core edges
    G.add_edge("Customer", "Account",     etype="owns")
    G.add_edge("Account", "Transaction",  etype="makes")
    G.add_edge("Transaction", "Merchant", etype="paid_to")
    G.add_edge("Account", "Device",       etype="uses")

    # Peer accounts if device is shared
    ring = device_accounts > 3
    n_peers = min(device_accounts - 1, 5)  # cap display at 5 peer accounts
    for i in range(n_peers):
        peer_id = f"PeerAcct{i+1}"
        fraud_peer = ring and (i < n_peers // 2)
        ntype = "fraud_txn" if fraud_peer else "peer_acct"
        lbl = f"Peer Acct {i+1}\n{'⚠ high fraud' if fraud_peer else 'normal'}"
        G.add_node(peer_id, ntype=ntype, label=lbl)
        G.add_edge(peer_id, "Device", etype="uses")

    # Layout
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Manual positions for clarity
    pos = {
        "Customer":    (-2.5, 0.5),
        "Account":     (-1.2, 0.0),
        "Transaction": (0.0,  0.0),
        "Merchant":    (1.3,  0.0),
        "Device":      (-1.2, -1.5),
    }
    # Arrange peer accounts in an arc below the device
    angles = [math.pi + math.pi * i / max(n_peers, 1) for i in range(n_peers)]
    for i in range(n_peers):
        peer_id = f"PeerAcct{i+1}"
        px = -1.2 + 1.2 * math.cos(angles[i])
        py = -1.5 + 0.7 * math.sin(angles[i]) - 0.6
        pos[peer_id] = (px, py)

    # Node colors
    node_color_list = [
        NODE_COLORS.get(G.nodes[n].get("ntype", "transaction"), "#888")
        for n in G.nodes()
    ]
    node_size_list = [
        2000 if n == "Transaction" else 1200
        for n in G.nodes()
    ]

    # Draw ring edges with different colour
    ring_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if G.nodes[u].get("ntype") in ("peer_acct", "fraud_txn")
    ]
    normal_edges = [e for e in G.edges() if e not in ring_edges]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                           edge_color=EDGE_COLOR, arrows=True, ax=ax,
                           arrowsize=15, width=1.5,
                           connectionstyle="arc3,rad=0.1")
    if ring_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ring_edges,
                               edge_color=RING_EDGE_COLOR, arrows=True, ax=ax,
                               arrowsize=12, width=1.2, style="dashed",
                               connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color_list,
                           node_size=node_size_list,
                           ax=ax, alpha=0.95)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=7, font_color="white", ax=ax)

    # Edge labels
    edge_labels = {
        ("Customer", "Account"): "owns",
        ("Account", "Transaction"): "makes",
        ("Transaction", "Merchant"): "paid_to",
        ("Account", "Device"): "uses",
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="#cccccc", ax=ax)

    # Legend
    patches = [
        mpatches.Patch(color=NODE_COLORS["customer"],    label="Customer"),
        mpatches.Patch(color=NODE_COLORS["account"],     label="Account (target)"),
        mpatches.Patch(color=NODE_COLORS["transaction"], label="Transaction (scored)"),
        mpatches.Patch(color=NODE_COLORS["merchant"],    label="Merchant"),
        mpatches.Patch(color=NODE_COLORS["device"],      label="Device"),
        mpatches.Patch(color=NODE_COLORS["peer_acct"],   label="Peer Account"),
        mpatches.Patch(color=NODE_COLORS["fraud_txn"],   label="High-fraud Peer"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              facecolor="#1E1E1E", labelcolor="white", edgecolor="#444")

    ring_note = " — SHARED DEVICE RING DETECTED" if ring else ""
    ax.set_title(
        f"Conceptual local graph (entered values){ring_note}",
        color="white", fontsize=9, pad=8
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


def _draw_ring_fraud_diagram() -> plt.Figure:
    """Static ring-fraud teaching diagram: one device, many accounts."""
    G = nx.DiGraph()
    G.add_node("Device X",     ntype="device")
    G.add_node("Account A\n(target)",  ntype="account")
    G.add_node("Account B\n(fraud)",   ntype="fraud_txn")
    G.add_node("Account C\n(fraud)",   ntype="fraud_txn")
    G.add_node("Account D\n(fraud)",   ntype="fraud_txn")
    G.add_node("Txn: $450\nNORMAL",    ntype="transaction")
    G.add_node("Txn: fraud\n(known)",  ntype="fraud_txn")
    G.add_node("Txn: fraud\n(known) ", ntype="fraud_txn")

    G.add_edge("Account A\n(target)",  "Device X",            etype="uses")
    G.add_edge("Account B\n(fraud)",   "Device X",            etype="uses")
    G.add_edge("Account C\n(fraud)",   "Device X",            etype="uses")
    G.add_edge("Account D\n(fraud)",   "Device X",            etype="uses")
    G.add_edge("Account A\n(target)",  "Txn: $450\nNORMAL",   etype="makes")
    G.add_edge("Account B\n(fraud)",   "Txn: fraud\n(known)", etype="makes")
    G.add_edge("Account C\n(fraud)",   "Txn: fraud\n(known) ",etype="makes")

    pos = {
        "Device X":             (0.0,  0.0),
        "Account A\n(target)":  (-2.0, 1.0),
        "Account B\n(fraud)":   (-0.8, 1.5),
        "Account C\n(fraud)":   (0.8,  1.5),
        "Account D\n(fraud)":   (2.0,  1.0),
        "Txn: $450\nNORMAL":    (-3.0, 2.2),
        "Txn: fraud\n(known)":  (-0.8, 2.8),
        "Txn: fraud\n(known) ": (0.8,  2.8),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    nc = [NODE_COLORS.get(G.nodes[n].get("ntype", "account"), "#888") for n in G.nodes()]
    ns = [1600 if "Device" in n else 1200 for n in G.nodes()]
    nx.draw_networkx(G, pos, node_color=nc, node_size=ns,
                     font_size=7, font_color="white",
                     edge_color=EDGE_COLOR, arrows=True, ax=ax)
    ax.annotate(
        "Tabular model: sees only Txn: $450 features → misses ring\n"
        "GNN: propagates from Device X → Account A → Transaction → flags it",
        xy=(0, -0.6), xycoords="data", color="#FFD700",
        ha="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A2E", edgecolor="#FFD700")
    )
    ax.set_title("Ring Fraud: one device shared across accounts", color="white", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SHARED: build score payload from fields
# ═════════════════════════════════════════════════════════════════════════════

def _build_payload(
    amount, avg_amount, hour, is_cross_border, merchant_risk,
    balance, velocity, acct_age, cust_age, tenure, country_risk,
    device_accounts, device_customers, device_risk,
    low_thresh=0.30, high_thresh=0.70,
) -> dict:
    zscore = (amount - avg_amount) / max(avg_amount * 0.4, 1.0)
    return {
        "amount": amount,
        "amount_zscore": round(zscore, 3),
        "hour_of_day": hour,
        "is_cross_border": int(is_cross_border),
        "time_since_last_txn": 86400.0,
        "merchant_risk_score": merchant_risk,
        "merchant_category_enc": 0,
        "balance": balance,
        "avg_transaction_amount": avg_amount,
        "transaction_velocity": velocity,
        "account_age_days": acct_age,
        "age": float(cust_age),
        "tenure_days": float(tenure),
        "country_risk_score": float(country_risk),
        "num_accounts": 1,
        "merchant_transaction_count": 100,
        "merchant_country_risk": 1.0,
        "device_num_accounts": device_accounts,
        "device_num_customers": device_customers,
        "device_risk_score": device_risk,
        "low_threshold": low_thresh,
        "high_threshold": high_thresh,
    }


def _call_score(payload: dict) -> dict:
    try:
        r = requests.post(f"{BACKEND}/gnn/score", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _render_score_result(result: dict, context_label: str = "") -> None:
    """Render score + decision + explanation in a compact layout."""
    if "error" in result:
        st.error(f"Backend error: {result['error']}")
        return
    if "detail" in result:
        st.error(f"Backend: {result['detail']}")
        return

    score = result.get("fraud_risk_score", 0.0)
    decision = result.get("decision", "?")

    icons = {"APPROVE": "✅", "REVIEW": "⚠️", "BLOCK": "🚫"}
    colors = {"APPROVE": "#27AE60", "REVIEW": "#F39C12", "BLOCK": "#E74C3C"}
    icon = icons.get(decision, "❓")
    color = colors.get(decision, "#888")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Fraud Risk Score", f"{score:.4f}",
                  help="Uncalibrated model score — not a fraud probability.")
        st.markdown(
            f"<div style='background:{color};padding:8px 14px;border-radius:6px;"
            f"font-size:1.2rem;font-weight:bold;color:white;text-align:center'>"
            f"{icon} {decision}</div>",
            unsafe_allow_html=True,
        )
        policy = result.get("policy", {})
        st.caption(
            f"Policy: < {policy.get('low_threshold','0.30')} = APPROVE | "
            f"< {policy.get('high_threshold','0.70')} = REVIEW | "
            f"else = BLOCK"
        )
        if context_label:
            st.caption(f"Scenario: *{context_label}*")
    with c2:
        st.progress(min(float(score), 1.0), text=f"Fraud Risk: {score:.1%}")
        st.caption(result.get("narrative", ""))

    nodes = result.get("important_nodes", [])
    if nodes:
        with st.expander("Important nodes & features", expanded=False):
            for n in nodes:
                st.write(
                    f"• **{n['node_type']}** — "
                    f"`{n['top_feature']}` "
                    f"(attribution {n['attribution_score']:.5f})"
                )
            st.caption(
                "Attribution = |gradient × input|. This shows which node features "
                "the model gave most weight to. It is model evidence, not causal proof."
            )

    caveats = result.get("caveats", [])
    if caveats:
        with st.expander("Caveats", expanded=False):
            for c in caveats:
                st.caption(f"⚠ {c}")
            st.caption(result.get("warning", ""))


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
(
    tab_test,
    tab_scenarios,
    tab_graph,
    tab_why,
    tab_how,
    tab_model,
) = st.tabs([
    "🔍 Test Transaction",
    "🧪 Teaching Scenarios",
    "🗺️ Graph Explorer",
    "⚖️ Why GNN?",
    "⚙️ How It Works",
    "📋 Model & Limitations",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Test Transaction
# ─────────────────────────────────────────────────────────────────────────────
with tab_test:
    # ── Scenario presets ──────────────────────────────────────────────────────
    PRESETS: dict[str, dict] = {
        "Normal Transaction": {
            "amount": 500.0, "avg_amount": 400.0, "hour": 14,
            "is_cross_border": False, "merchant_risk": 0.05,
            "balance": 8000.0, "velocity": 2.0, "acct_age": 365.0,
            "cust_age": 35, "tenure": 730, "country_risk": 1,
            "device_accounts": 1, "device_customers": 1, "device_risk": 0.04,
        },
        "Suspicious Device Ring": {
            "amount": 380.0, "avg_amount": 350.0, "hour": 13,
            "is_cross_border": False, "merchant_risk": 0.06,
            "balance": 4500.0, "velocity": 3.0, "acct_age": 200.0,
            "cust_age": 28, "tenure": 400, "country_risk": 1,
            "device_accounts": 10, "device_customers": 8, "device_risk": 0.82,
        },
        "High-Risk Merchant": {
            "amount": 750.0, "avg_amount": 600.0, "hour": 10,
            "is_cross_border": False, "merchant_risk": 0.88,
            "balance": 6000.0, "velocity": 2.0, "acct_age": 400.0,
            "cust_age": 40, "tenure": 1000, "country_risk": 1,
            "device_accounts": 1, "device_customers": 1, "device_risk": 0.05,
        },
        "Account Takeover": {
            "amount": 12000.0, "avg_amount": 900.0, "hour": 2,
            "is_cross_border": True, "merchant_risk": 0.05,
            "balance": 50000.0, "velocity": 12.0, "acct_age": 1200.0,
            "cust_age": 52, "tenure": 3650, "country_risk": 1,
            "device_accounts": 1, "device_customers": 1, "device_risk": 0.03,
        },
        "Start From Scratch": {
            "amount": 500.0, "avg_amount": 400.0, "hour": 12,
            "is_cross_border": False, "merchant_risk": 0.05,
            "balance": 10000.0, "velocity": 2.0, "acct_age": 365.0,
            "cust_age": 35, "tenure": 730, "country_risk": 1,
            "device_accounts": 1, "device_customers": 1, "device_risk": 0.05,
        },
    }
    PRESET_HINTS: dict[str, str] = {
        "Normal Transaction": (
            "Routine domestic payment, private device, normal hour. "
            "Expected: low risk from both tabular and GNN models."
        ),
        "Suspicious Device Ring": (
            "Transaction features look completely normal — but the device is shared with 10 accounts. "
            "This is the GNN's core advantage: a tabular model misses this entirely."
        ),
        "High-Risk Merchant": (
            "Transaction to a merchant flagged as high-risk (0.88). "
            "Tests whether the merchant signal alone changes the decision."
        ),
        "Account Takeover": (
            "Large amount at 2 AM, cross-border, velocity = 12. "
            "Classic ATO pattern — strong transaction-level signals without ring context."
        ),
        "Start From Scratch": (
            "All defaults reset. Build your own transaction from scratch "
            "to explore individual signal effects."
        ),
    }

    # ── Model not loaded: setup card ──────────────────────────────────────────
    if not model_ready:
        st.error(
            "**GNN model not loaded — live scoring is disabled.**\n\n"
            "To enable scoring, complete these four steps:\n\n"
            "1. Open `notebooks/04_gnn_fraud_detection.ipynb` in Jupyter\n"
            "2. Run all cells end-to-end (takes ~2 minutes on CPU)\n"
            "3. Confirm the final cell prints: `Model saved to models/gnn_fraud_detector.pt`\n"
            "4. Restart the backend: `python -m uvicorn main:app --app-dir backend --port 8000`\n\n"
            "The **Graph Explorer**, **Why GNN?**, **How It Works**, and **Model & Limitations** "
            "tabs are fully available without a loaded model."
        )
        if st.button("🔄 Check Model Again", key="check_model_again"):
            st.rerun()
        st.divider()

    # ── Preset selector ───────────────────────────────────────────────────────
    if "gnn_form_version" not in st.session_state:
        st.session_state.gnn_form_version = 0
    if "gnn_preset_vals" not in st.session_state:
        st.session_state.gnn_preset_vals = PRESETS["Normal Transaction"].copy()

    ps_col, ph_col = st.columns([2, 3])
    with ps_col:
        preset_choice = st.selectbox(
            "Scenario preset",
            list(PRESETS.keys()),
            help="Load a pre-built scenario to explore a specific fraud pattern, then tweak values.",
        )
    with ph_col:
        st.caption(PRESET_HINTS[preset_choice])
        if st.button("⬇ Load preset into form", key="load_preset"):
            st.session_state.gnn_preset_vals = PRESETS[preset_choice].copy()
            st.session_state.gnn_form_version += 1
            st.rerun()

    p = st.session_state.gnn_preset_vals

    # ── Field Guide ───────────────────────────────────────────────────────────
    with st.expander("❓ Field Guide & Glossary — click to expand", expanded=False):
        _h_fields, _h_scores, _h_why, _h_graph, _h_test, _h_gloss = st.tabs([
            "📋 Field Guide",
            "🎯 Scores & Decisions",
            "⚖️ Why GNN?",
            "🕸️ The Graph",
            "🧪 How to Test",
            "📖 Glossary",
        ])

        # ── Tab: Field Guide ──────────────────────────────────────────────────
        with _h_fields:
            st.markdown("### Every input field explained")
            st.caption(
                "The form is split into three sections: "
                "**A — Transaction**, **B — Account & Customer**, and "
                "**C — Network / Graph Context**."
            )

            st.markdown("---")
            st.markdown("#### A — Transaction fields")

            st.markdown("""
**Amount ($)**

| Item | Detail |
|------|--------|
| What it is | The value being transferred or paid |
| Real-world meaning | The monetary amount of one payment or transfer |
| Why the model needs it | Large amounts can be a fraud signal, especially relative to account history |
| Used by | Tabular baseline + GNN |
| Typical range | $1 – $50,000 (synthetic demo; depends on account type) |
| If high | May raise risk score, especially if unusually large for this account |
| If low | Generally lower suspicion — but not sufficient alone to clear a transaction |
| Example | $450 for groceries; $8,000 for an electronics purchase |

> Amount alone does not determine fraud. The model also considers
> whether the amount is unusual *relative to this account's history* (the Z-score below).

---

**Account average transaction ($)**

| Item | Detail |
|------|--------|
| What it is | The account's typical transaction amount over its history |
| Real-world meaning | An account's "normal spending level" |
| Why the model needs it | Used to compute the **Amount Z-score** automatically |
| Used by | Tabular baseline + GNN (via Z-score) |
| Typical range | $100 – $5,000 depending on account type |
| If close to Amount | Z-score ≈ 0 → transaction is typical for this account |
| If much lower than Amount | Z-score > 0 → this is a large transaction relative to history |
| Example | Account normally spends $400; today's $3,000 transaction is unusual |

> **Derived value — Amount Z-score:**
> The app computes this automatically using:
> ```
> amount_zscore = (amount − account_average) / (account_average × 0.4)
> ```
> The divisor `account_average × 0.4` is a fixed approximation of the account's
> spending variability (standard deviation proxy).
> - Z-score near 0 → close to normal historical behaviour
> - Z-score positive → larger than usual
> - Z-score negative → smaller than usual
> - |Z-score| > 2 → notably unusual relative to account history
>
> You do not enter the Z-score directly — it is computed from Amount and Account Average.

---

**Hour of day (0–23)**

| Item | Detail |
|------|--------|
| What it is | The hour when the transaction occurs (24-hour clock) |
| Real-world meaning | Time of day as a fraud context signal |
| Why the model needs it | Fraud transactions tend to cluster at unusual hours (very late night / early morning) |
| Used by | Tabular baseline + GNN |
| Typical range | 8–20 for routine transactions |
| Encoded as | `hour_sin` and `hour_cos` — cyclic encoding so 23:00 and 00:00 are adjacent |
| If unusual (0–5, 22–23) | Raises risk, especially combined with large amount or cross-border |
| Example | 2 AM domestic transfer of $12,000 — unusual hour + large amount combination |

> The model uses **cyclic encoding** of the hour: `sin(2π × hour / 24)` and
> `cos(2π × hour / 24)`. This preserves the circular nature of time (23:00 and 00:00
> are one hour apart, not 23 hours apart).

---

**Cross-border transaction**

| Item | Detail |
|------|--------|
| What it is | Whether the transaction crosses national boundaries |
| Real-world meaning | International transfers carry higher contextual risk in fraud detection |
| Why the model needs it | Cross-border is a known fraud-ring signal when combined with other suspicious features |
| Used by | Tabular baseline + GNN |
| If checked | Increases contextual risk, especially at unusual hours or with large amounts |
| If unchecked | Domestic transaction — lower base risk from this signal |
| Example | A domestic £200 grocery payment vs. a 3 AM €9,000 transfer to an overseas account |

---

**Merchant risk score**

| Item | Detail |
|------|--------|
| What it is | A pre-computed risk score for the merchant receiving the payment (0 = safe, 1 = high-risk) |
| Real-world meaning | Some merchants are known fraud magnets — they repeatedly appear in fraudulent transactions |
| Why the model needs it | Direct signal about the payment recipient's history; also propagates through the graph |
| Used by | Tabular baseline (directly) + GNN (also via merchant neighbourhood) |
| Typical range | 0.01 – 0.15 for normal merchants; 0.70 – 0.95 for planted high-risk merchants |
| If high | Raises risk directly — and the GNN aggregates this across all transactions to that merchant |
| If low (0.05) | Safe merchant — little direct risk from this signal |
| Example | 0.05 for a known supermarket chain; 0.88 for a merchant flagged in many past fraud cases |

            """)

            st.markdown("---")
            st.markdown("#### B — Account & Customer fields")

            st.markdown("""
**Account balance ($)**

| Item | Detail |
|------|--------|
| What it is | The account's current available balance |
| Real-world meaning | How much money is in the account right now |
| Why the model needs it | An amount-to-balance ratio can signal whether a transaction is proportionally large |
| Used by | GNN (account node feature) |
| Typical range | $500 – $100,000 |
| If low relative to amount | Transaction is a large fraction of the account's funds — contextually suspicious |
| Example | Balance $500, transaction $450 — nearly emptying the account |

---

**Account velocity (txns/day)**

| Item | Detail |
|------|--------|
| What it is | Average number of transactions per day for this account (rolling average) |
| Real-world meaning | How frequently this account transacts |
| Why the model needs it | A sudden burst of many transactions in a short window (velocity burst) is a fraud pattern |
| Used by | Tabular baseline (indirectly via `time_since_last_txn`) + GNN (account node feature) |
| Typical range | 0.5 – 5.0 for normal accounts |
| If high (> 10) | Velocity burst — many transactions in rapid succession — Fraud Pattern D |
| Example | 2.0 = normal (one transaction every 12 hours on average); 20.0 = suspicious burst |

> **Note on `time_since_last_txn`:** This field is fixed at 86,400 seconds (24 hours)
> in the current demo. It is not a user-configurable field. The velocity field
> (`txns/day`) is the primary control for velocity-related context.

---

**Account age (days)**

| Item | Detail |
|------|--------|
| What it is | How many days since this bank account was opened |
| Real-world meaning | Newly created accounts are a higher fraud risk — fraudsters open accounts specifically to conduct fraud |
| Used by | GNN (account node feature) |
| Typical range | 30 – 3,650 (1 month to 10 years) |
| If low (< 90 days) | Newly opened — higher base risk, especially if combined with suspicious device |
| Example | 30 days = brand-new account; 730 days = 2-year-old established account |

---

**Customer age**

| Item | Detail |
|------|--------|
| What it is | The account holder's age in years |
| Real-world meaning | Demographic context — younger customers can have different fraud profiles |
| Used by | GNN (customer node feature) |
| Typical range | 18 – 80 |
| Example | 35 = typical adult customer |

---

**Customer tenure (days)**

| Item | Detail |
|------|--------|
| What it is | How many days since this customer relationship began with the bank |
| Real-world meaning | Long-tenured customers are generally lower fraud risk — fraudsters don't build long relationships |
| Used by | GNN (customer node feature) |
| Typical range | 30 – 5,000+ |
| If low (< 180 days) | New customer relationship — higher base risk |
| Example | 730 = 2-year customer; 3,650 = 10-year customer |

---

**Customer country risk (1=low, 4=high)**

| Item | Detail |
|------|--------|
| What it is | A categorical risk score (1–4) for the customer's registered country |
| Real-world meaning | In banking risk models, some countries have higher statistical fraud rates |
| Used by | GNN (customer node feature) |
| Values | 1 = low risk; 2 = moderate; 3 = elevated; 4 = high risk |
| Example | 1 for most domestic customers; 4 for high-risk jurisdictions |

> ⚠ **Important caveat:** Country risk = 4 in this **synthetic demonstration** means the
> model was trained with that country associated with fraud patterns. This is a
> **model input feature in a teaching demo** — it is NOT a real-world judgment
> about people from any country, and it does NOT represent actual banking policy.
> In a real system, country-risk classification would be subject to strict
> anti-discrimination and regulatory requirements.

            """)

            st.markdown("---")
            st.markdown("#### C — Network / Graph Context fields (the GNN differentiator)")

            st.markdown("""
**Accounts sharing this device**

| Item | Detail |
|------|--------|
| What it is | Number of bank accounts that have been accessed using this device fingerprint |
| Real-world meaning | Fraudsters often control many accounts accessed from the same device (a "fraud ring") |
| Why the model needs it | The single strongest relational signal — multiple unrelated accounts on one device is a near-definitive ring indicator |
| Used by | GNN (device node feature — the core ring detection signal) |
| Typical range | 1 (private device) to 20+ (ring device) |
| If 1 | Private device — no ring signal from this field |
| If > 3 | Suspicious — graph visualization will show peer accounts |
| If > 8 | Strongly suspicious in the synthetic model |
| Example | 1 = your personal phone; 12 = a device used by 12 different account holders |

> This is the primary way to observe the GNN's **ring fraud detection**
> advantage. Set this to 10 and Device risk to 0.85 to see the effect.

---

**Customers sharing this device**

| Item | Detail |
|------|--------|
| What it is | Number of distinct customers (not just accounts) who have used this device |
| Real-world meaning | One customer with multiple accounts on one device is less suspicious than multiple unrelated customers on one device |
| Used by | GNN (device node feature) |
| Typical range | 1 (private) to 10+ (ring) |
| Example | 1 = owned by one person; 8 = used by 8 different people |

---

**Device risk score**

| Item | Detail |
|------|--------|
| What it is | A pre-computed aggregated risk score for this device (0 = safe, 1 = high-risk) |
| Real-world meaning | Devices that have been involved in many fraudulent transactions accumulate a high risk score |
| Used by | GNN (device node feature) |
| Typical range | 0.01 – 0.10 for normal devices; 0.70 – 0.95 for ring devices |
| If high (> 0.70) | Strong ring signal — this device is associated with fraud history |
| Example | 0.04 = clean personal device; 0.85 = known ring device in the synthetic dataset |

            """)

            st.markdown("---")
            st.markdown("#### ⚙️ Advanced Decision Policy (thresholds)")

            st.markdown("""
These are inside the collapsible **⚙️ Advanced Decision Policy** section of the form.

**Review threshold** (default: 0.30)

| Item | Detail |
|------|--------|
| What it is | The Fraud Risk Score above which a transaction goes into **REVIEW** instead of **APPROVE** |
| Real-world meaning | The sensitivity dial — lower = catch more fraud but more false alarms |
| Default | 0.30 (score ≥ 0.30 → REVIEW) |

**Block threshold** (default: 0.70)

| Item | Detail |
|------|--------|
| What it is | The Fraud Risk Score above which a transaction is **BLOCKED** |
| Real-world meaning | The high-confidence dial — only block when score is very high |
| Default | 0.70 (score ≥ 0.70 → BLOCK) |

> These are **demo defaults**, not universal standards. Real systems calibrate thresholds
> against the cost of false positives (blocking legitimate payments) vs. false negatives (allowing fraud).

            """)

            st.markdown("---")
            st.markdown("#### 🔒 Fixed internal fields (not user-configurable)")
            st.markdown("""
The following fields are sent to the backend but are fixed in the demo UI.
They are listed here so you know they exist:

| Field | Fixed value | What it means |
|-------|-------------|---------------|
| `time_since_last_txn` | 86,400 seconds (24 hours) | Time since the previous transaction on this account |
| `merchant_category_enc` | 0 | Encoded merchant category integer (not exposed in demo UI) |
| `num_accounts` | 1 | Number of accounts owned by this customer |
| `merchant_transaction_count` | 100 | Number of total transactions at this merchant |
| `merchant_country_risk` | 1.0 | Country risk for the merchant's location |

These are reasonable defaults for the demo. In the full training pipeline,
they are populated from the synthetic dataset.
            """)

        # ── Tab: Scores & Decisions ───────────────────────────────────────────
        with _h_scores:
            st.markdown("### 🎯 What does the Fraud Risk Score mean?")
            st.markdown("""
**Fraud Risk Score ∈ [0, 1]**

This is the model's output indicating how strongly it associates the
transaction with the synthetic fraud patterns it learned during training.

It is **not** a calibrated real-world probability of fraud.

| Score | Interpretation |
|-------|---------------|
| 0.05  | Relatively low model risk — transaction and context look normal |
| 0.30  | Moderate — some suspicious signals; review threshold |
| 0.45  | Intermediate — multiple signals present |
| 0.70  | High — strong fraud signals; block threshold |
| 0.85  | Relatively high model risk — strong association with fraud patterns |

These are **illustrative examples**, not fixed thresholds.

> The model was trained on **synthetic data** with a ~5.8% fraud rate.
> A score of 0.70 does **not** mean a 70% probability of real fraud.
> It means the model's computation strongly matches the synthetic fraud patterns.
            """)

            st.markdown("---")
            st.markdown("### 🚦 What does the decision mean?")
            st.markdown("""
The Fraud Risk Score is converted to a decision using configurable thresholds:

```
Fraud Risk Score
       │
       ├── score < 0.30  ─────────→  ✅  APPROVE
       │                              Transaction proceeds
       │
       ├── 0.30 ≤ score < 0.70  ──→  ⚠️  REVIEW
       │                              Queued for analyst review
       │
       └── score ≥ 0.70  ─────────→  🚫  BLOCK
                                      Transaction rejected
```

**Default thresholds:** Review at 0.30, Block at 0.70.
These are adjustable using the threshold sliders in the form.

**Why configurable?**
The right operating point depends entirely on context:
- A bank with high fraud losses might lower the Review threshold to 0.15
- A low-risk merchant platform might only block at 0.90
- There is no single "correct" threshold — it is a business decision

**These thresholds are illustrative defaults for this demonstration.**
Real systems calibrate against the cost of a false positive (blocking a
legitimate transaction) versus a false negative (allowing fraud through).
            """)

            st.markdown("---")
            st.markdown("### ⚙️ What do the model architecture parameters mean?")
            st.info(
                "These are **model architecture parameters** — they describe how the GNN "
                "was built. You do not enter or change these values in the demo."
            )
            st.markdown("""
**GNN Layers = 2**

The number of message-passing rounds.
- Layer 1: each transaction node receives information from its direct neighbours
  (account, merchant)
- Layer 2: each transaction node now receives information from neighbours-of-neighbours
  (customer, device — via account)

Two layers is sufficient to reach the device node (the ring-fraud signal source)
without over-smoothing the graph.

---

**SAGEConv (GraphSAGE Convolution)**

The convolution operator used in each HeteroConv layer.
SAGEConv computes each node's new representation as:
```
h_v = W · concat(h_v,  mean(h_u for u in neighbours(v)))
```
It combines the node's own features with the mean of its neighbours' features.
Chosen because our transaction graph is sparse and well-structured — each
transaction touches exactly one account and one merchant.

---

**HeteroConv (Heterogeneous Convolution)**

A wrapper that applies a different SAGEConv per edge type, then sums the
contributions at each destination node. Used because our graph has multiple
relationship types (owns, makes, paid_to, uses) that carry different meanings.

---

**Hidden dimension = 64**

All node types are projected into a 64-dimensional embedding space before
message passing. This common dimension allows different node types to
exchange information during message passing.

---

**Dropout = 0.3**

30% of neuron activations are randomly zeroed during training.
Prevents the model from memorising the training data.
Applied after each message-passing layer and in the classification head.

---

**Residual connections**

After each message-passing layer, the layer's output is added back to its
input (`h_new = h_new + h_prev`). This prevents over-smoothing — without
residuals, all node embeddings can converge to similar values as depth increases.

---

**MLP Classification Head**

After two rounds of message passing, the transaction node's 64-dim embedding
is passed through a small neural network:
```
Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1) → sigmoid
```
This produces the final Fraud Risk Score.

---

**Total parameters: ~170,241**

Small by modern deep learning standards — appropriate for this graph size.
            """)

        # ── Tab: Why GNN? ─────────────────────────────────────────────────────
        with _h_why:
            st.markdown("### Traditional ML vs. GNN")
            st.markdown("""
**Traditional ML** looks at each transaction in isolation:

```
Transaction
     │
     ├── Amount
     ├── Time of day
     ├── Cross-border flag
     ├── Merchant risk score
     └── Account velocity
          │
          ▼
       Model (Logistic Regression / LightGBM)
          │
          ▼
      Risk Score
```

It asks: *"Does this single row look suspicious?"*

---

**GNN** additionally considers the connected entities:

```
                    Customer
                    (age, tenure, country risk)
                       │ owns
                    Account
                    (balance, velocity, age)
                    /          \\
              uses               makes
                /                  \\
           Device              Transaction  ───paid_to───►  Merchant
      (num_accounts,            (amount,                   (risk score,
       num_customers,            hour,                      txn count)
       device_risk)              cross_border)
           │
     Other Accounts
     on same Device
           │
     Their Transactions
           │
           ▼
      GNN Message Passing (2 layers)
           │
           ▼
      Risk Score (enriched with neighbourhood context)
```

It asks: *"Does this transaction look suspicious given its own features
**and** what it is connected to?"*

---

**The key educational claim:**

> Fraud Pattern B (ring fraud) has **normal transaction features**.
> A tabular model assigns low risk. The GNN may assign higher risk
> because it can "see" that the account's device is shared with
> many other accounts that have fraud histories.
            """)

        # ── Tab: The Graph ────────────────────────────────────────────────────
        with _h_graph:
            st.markdown("### 🕸️ Understanding the graph")
            st.markdown("""
The GNN operates on a **heterogeneous graph** — a graph with multiple types
of nodes and multiple types of edges.

#### Nodes

| Node | What it represents | Features |
|------|--------------------|---------|
| 🧑 **Customer** | The human account holder | age, tenure, country risk, #accounts |
| 🏦 **Account** | A bank account owned by a customer | balance, avg amount, velocity, age |
| 💳 **Transaction** | A single payment or transfer ← **fraud label lives here** | amount, z-score, hour, cross-border, velocity, merchant risk, category |
| 🏪 **Merchant** | The entity receiving the payment | risk score, #transactions, country risk |
| 📱 **Device** | Device fingerprint / IP cluster used to access accounts | #accounts sharing, #customers sharing, risk score |

#### Edges (relationships)

| Edge | Direction | Why it exists |
|------|-----------|---------------|
| `owns` | Customer → Account | Customer identity context flows to account |
| `makes` | Account → Transaction | Account history informs transaction risk |
| `paid_to` | Transaction → Merchant | Merchant fraud history propagates back |
| `uses` (cust level) | Customer → Device | Device sharing at customer level |
| `uses` (acct level) | Account → Device | Device sharing at account level — stronger ring signal |

#### Why reverse edges?

Each forward edge has a **reverse edge**:
`owns` → `rev_owns`, `makes` → `rev_makes`, etc.

Without reverse edges, information only flows one way. With reverse edges,
the account node can also receive signals from *its* transactions
(e.g., "one of my transactions is cross-border at 3 AM").

#### The shared device — why it matters

```
Customer A → Account A ──→  Device X  ←── Account B ← Customer B
                                ↑
                           Account C ← Customer C
                                ↑
                           Account D ← Customer D (known fraud)
```

A transaction from Account A looks normal in isolation. But through
`Account A → uses → Device X`, the GNN discovers that Device X is
shared with Account D, which has a fraud history. This 2-hop path
is only reachable through the graph — a tabular model cannot see it.
            """)

        # ── Tab: How to Test ──────────────────────────────────────────────────
        with _h_test:
            st.markdown("### 🧪 How to test this demo")
            st.markdown("""
Use the **scenario preset** selector at the top of the form to quickly load a specific fraud pattern, then adjust individual values. Each preset pre-fills all 15 fields.

---

#### Preset: Normal Transaction
Load this preset and click **🔍 Analyze Transaction**. All features and graph context are benign.
**Expected:** Low risk from both tabular and GNN models.

---

#### Preset: Suspicious Device Ring
Load this preset and click Analyze. Transaction features are completely normal (amount $380, hour 13, no cross-border) — but `device_accounts = 10`, `device_risk = 0.82`.
**Expected lesson:** The GNN should assign higher risk than a tabular model would, because it can see the shared-device ring context.

---

#### Preset: High-Risk Merchant
Merchant risk = 0.88 — a known high-risk merchant. All other features are normal.
**Expected lesson:** Does a single high-risk merchant signal change the decision?

---

#### Preset: Account Takeover
Amount = $12,000, hour = 2 AM, cross-border = Yes, velocity = 12. Classic ATO pattern.
**Expected lesson:** Strong transaction-level signals. Clean device. Traditional ML flags this; GNN adds context.

---

#### Experiment: isolate the ring signal
1. Load **Normal Transaction** → Analyze → note the score
2. Only change: **Accounts sharing device → 10**, **Device risk → 0.85**
3. Analyze again → does the score change from graph context alone?

---

#### Experiment: what changes the score?

| Change | What to observe |
|--------|----------------|
| Amount = 5× account average | Z-score rises → risk may increase |
| Hour = 2 or 3 | Unusual time signal |
| Enable cross-border | Adds contextual risk |
| Velocity = 15+ | Velocity burst pattern |
| Set account average = amount | Z-score → 0; risk may drop |
| Device accounts = 10, device risk = 0.85 | Ring context — GNN-specific signal |
| Merchant risk = 0.88 | High-risk merchant direct signal |

> The model sees all features simultaneously. Changing one signal may not produce
> a dramatic shift if other signals remain neutral.
            """)

        # ── Tab: Glossary ─────────────────────────────────────────────────────
        with _h_gloss:
            st.markdown("### 📖 Terminology glossary")
            st.markdown("""
**Node**
A single entity in the graph — a customer, account, transaction, merchant, or device.
Each node has a set of numerical features.

**Edge**
A connection between two nodes representing a relationship — e.g., "Account A *makes*
Transaction T" or "Account A *uses* Device X". Edges carry no features in this model
(features are on nodes).

**Heterogeneous Graph**
A graph with more than one type of node and/or edge. Our graph has 5 node types and
10 edge types (5 forward + 5 reverse), which is why it is "heterogeneous".

**Message Passing**
The core GNN operation: each node collects information from its connected neighbours
and updates its own representation by combining its own features with the aggregated
neighbourhood signal. Repeated over 2 layers.

**Neighbourhood**
The set of nodes directly connected to a given node (1-hop neighbourhood).
The 2-hop neighbourhood includes nodes two edges away.

**Node Features**
The numerical values stored at each node — e.g., `amount`, `hour_sin` for a transaction
node or `num_accounts`, `device_risk_score` for a device node.

**Embedding**
A dense vector representation of a node after the input projection and message-passing
layers. All node types are projected into 64-dimensional embeddings.

**Fraud Risk Score**
The model's uncalibrated output score ∈ [0, 1] indicating how strongly the GNN
associates a transaction with the synthetic fraud patterns it learned. Not a probability.

**Baseline**
A conventional tabular ML model (Logistic Regression or LightGBM) trained on the same
data but without graph structure — used to measure what the GNN adds over standard ML.

**PR-AUC**
Precision-Recall Area Under the Curve. The primary evaluation metric, measuring
ranking quality across all decision thresholds. More informative than accuracy or
ROC-AUC on imbalanced datasets like this one (~5.8% fraud rate).

**Explainability / Attribution**
After scoring, the model computes `|gradient × input|` for each input feature to show
which features it weighted most heavily for this specific transaction. Called
"attribution" because it assigns credit to input features. Not causal proof.

**SAGEConv**
GraphSAGE convolution (Hamilton et al., 2017). Aggregates a node's own features with
the mean of its neighbours' features. Used here because the transaction graph is sparse
and well-structured.

**HeteroConv**
A wrapper in PyTorch Geometric that applies different learned convolution operators
for each edge type in a heterogeneous graph, then sums contributions per destination node.

**Transductive setting**
Training and test nodes are all present in the same graph during training — the model
sees the full graph structure even if test labels are withheld. Inflates metrics
compared to a cold-start (inductive) setting.
            """)
    # ── GNN Fraud Scenario Builder ────────────────────────────────────────────
    with st.form(f"gnn_score_{st.session_state.gnn_form_version}"):
        st.markdown("#### A — Transaction")
        a1, a2 = st.columns(2)
        with a1:
            amount = st.number_input(
                "Amount ($)",
                min_value=1.0, value=float(p["amount"]), step=10.0,
                help="Transaction value. The Z-score (amount vs. account average) is derived automatically.",
            )
            hour = st.slider(
                "Hour of day (0–23)", 0, 23, int(p["hour"]),
                help="24-hour clock. Unusual hours (0–5, 22–23) are a risk signal.",
            )
        with a2:
            avg_amt = st.number_input(
                "Account average transaction ($)",
                min_value=1.0, value=float(p["avg_amount"]), step=10.0,
                help="Account's typical spending level. Used to derive the amount Z-score — you don't enter Z-score directly.",
            )
            xcb = st.checkbox(
                "Cross-border transaction",
                value=bool(p["is_cross_border"]),
                help="International payments carry higher contextual risk.",
            )
        merch_r = st.slider(
            "Merchant risk score (0 = safe, 1 = high-risk)",
            0.0, 1.0, float(p["merchant_risk"]), 0.01,
            help="0.01–0.15 = normal merchant. 0.70–0.95 = known high-risk merchant in synthetic data.",
        )

        st.divider()
        st.markdown("#### B — Account & Customer")
        b1, b2, b3 = st.columns(3)
        with b1:
            balance = st.number_input(
                "Account balance ($)",
                min_value=0.0, value=float(p["balance"]),
                help="Current balance. GNN uses this as an account node feature.",
            )
            velocity = st.number_input(
                "Account velocity (txns/day)",
                min_value=0.0, value=float(p["velocity"]), step=0.5,
                help="Transactions per day. 10+ = velocity-burst fraud signal.",
            )
        with b2:
            acct_age = st.number_input(
                "Account age (days)",
                min_value=1.0, value=float(p["acct_age"]),
                help="New accounts (< 90 days) are a higher risk signal.",
            )
            cust_age = st.number_input(
                "Customer age",
                min_value=18, max_value=100, value=int(p["cust_age"]),
            )
        with b3:
            tenure = st.number_input(
                "Customer tenure (days)",
                min_value=1, value=int(p["tenure"]),
                help="Long tenure = lower base risk.",
            )
            c_risk = st.selectbox(
                "Customer country risk (1=low, 4=high)",
                [1, 2, 3, 4],
                index=int(p["country_risk"]) - 1,
            )

        st.divider()
        st.markdown("#### C — Network / Graph Context")
        st.caption("These fields drive the GNN's ring-detection advantage. A tabular model cannot see them.")
        c1, c2, c3 = st.columns(3)
        with c1:
            dev_accts = st.number_input(
                "Accounts sharing this device",
                min_value=1, value=int(p["device_accounts"]),
                help="1 = private device. 4+ = suspicious. 10+ = strong ring signal. The key GNN differentiator.",
            )
        with c2:
            dev_cust = st.number_input(
                "Customers sharing this device",
                min_value=1, value=int(p["device_customers"]),
                help="Multiple distinct people on one device = stronger ring signal.",
            )
        with c3:
            dev_risk = st.slider(
                "Device risk score (0 = clean, 1 = high-risk)",
                0.0, 1.0, float(p["device_risk"]), 0.01,
                help="0.01–0.10 = normal. 0.70–0.95 = ring device in synthetic data.",
            )

        with st.expander("⚙️ Advanced Decision Policy — thresholds"):
            tc1, tc2 = st.columns(2)
            with tc1:
                lo_thresh = st.slider(
                    "Review threshold", 0.0, 1.0, 0.30, 0.05,
                    help="Score ≥ this → REVIEW. Lower = more sensitive, more false alarms.",
                )
            with tc2:
                hi_thresh = st.slider(
                    "Block threshold", 0.0, 1.0, 0.70, 0.05,
                    help="Score ≥ this → BLOCK.",
                )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            disabled=not model_ready,
            type="primary",
        )

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        payload = _build_payload(
            amount, avg_amt, hour, xcb, merch_r,
            balance, velocity, acct_age, cust_age, tenure, c_risk,
            int(dev_accts), int(dev_cust), dev_risk, lo_thresh, hi_thresh,
        )
        with st.spinner("Analyzing via GNN…"):
            result = _call_score(payload)

        if "error" in result:
            st.error(f"Backend error: {result['error']}")
        elif "detail" in result:
            st.error(f"Backend: {result['detail']}")
        else:
            score = result.get("fraud_risk_score", 0.0)
            decision = result.get("decision", "?")
            _icons = {"APPROVE": "✅", "REVIEW": "⚠️", "BLOCK": "🚫"}
            _colors = {"APPROVE": "#27AE60", "REVIEW": "#F39C12", "BLOCK": "#E74C3C"}
            icon = _icons.get(decision, "❓")
            color = _colors.get(decision, "#888")

            st.subheader("Analysis Result")
            top_l, top_r = st.columns([1, 2])
            with top_l:
                st.metric(
                    "Fraud Risk Score", f"{score:.4f}",
                    help="Uncalibrated model score — not a real fraud probability.",
                )
                st.markdown(
                    f"<div style='background:{color};padding:10px 16px;border-radius:8px;"
                    f"font-size:1.3rem;font-weight:bold;color:white;text-align:center'>"
                    f"{icon} {decision}</div>",
                    unsafe_allow_html=True,
                )
                policy = result.get("policy", {})
                st.caption(
                    f"Thresholds: "
                    f"< {policy.get('low_threshold', lo_thresh)} APPROVE | "
                    f"< {policy.get('high_threshold', hi_thresh)} REVIEW | "
                    f"else BLOCK"
                )
            with top_r:
                st.progress(min(float(score), 1.0), text=f"Fraud Risk: {score:.1%}")
                st.caption(result.get("narrative", ""))

            st.divider()

            # Split: Transaction Signals | Graph / Network Signals
            sig_col, net_col = st.columns(2)
            zscore = (amount - avg_amt) / max(avg_amt * 0.4, 1.0)
            nodes = result.get("important_nodes", [])

            with sig_col:
                st.markdown("**Transaction Signals**")
                st.caption(
                    f"Amount: **${amount:,.2f}** "
                    f"(Z-score {zscore:+.2f} vs avg ${avg_amt:,.0f}) "
                    f"{'⚠ large vs history' if abs(zscore) > 2 else '✓ within normal range'}"
                )
                st.caption(
                    f"Hour: **{hour}:00** "
                    f"{'⚠ unusual hour' if hour < 6 or hour > 21 else '✓ normal hour'}"
                )
                st.caption(f"Cross-border: **{'Yes ⚠' if xcb else 'No ✓'}**")
                st.caption(
                    f"Merchant risk: **{merch_r:.2f}** "
                    f"{'⚠ high-risk merchant' if merch_r > 0.5 else '✓ normal merchant'}"
                )
                st.caption(
                    f"Velocity: **{velocity:.1f} txns/day** "
                    f"{'⚠ velocity burst' if velocity > 10 else '✓ normal'}"
                )
                txn_nodes = [n for n in nodes if n["node_type"] == "transaction"]
                if txn_nodes:
                    with st.expander("Top transaction-level attributions", expanded=False):
                        for n in txn_nodes:
                            st.write(f"• `{n['top_feature']}` — attribution {n['attribution_score']:.5f}")
                        st.caption("Attribution = |gradient × input|. Model evidence, not causal proof.")

            with net_col:
                st.markdown("**Graph / Network Signals**")
                ring_flag = int(dev_accts) > 3
                st.caption(
                    f"Device accounts: **{int(dev_accts)}** "
                    f"{'⚠ shared device — potential ring' if ring_flag else '✓ private device'}"
                )
                st.caption(
                    f"Device customers: **{int(dev_cust)}** "
                    f"{'⚠ multiple people' if int(dev_cust) > 2 else '✓ single owner'}"
                )
                st.caption(
                    f"Device risk score: **{dev_risk:.2f}** "
                    f"{'⚠ high-risk device' if dev_risk > 0.5 else '✓ clean device'}"
                )
                st.caption(
                    f"Account age: **{int(acct_age)} days** "
                    f"{'⚠ new account' if acct_age < 90 else '✓ established'}"
                )
                st.caption(
                    f"Customer tenure: **{int(tenure)} days** "
                    f"{'⚠ new relationship' if tenure < 180 else '✓ long-term'}"
                )
                graph_nodes = [n for n in nodes if n["node_type"] != "transaction"]
                if graph_nodes:
                    with st.expander("Top graph-level attributions", expanded=False):
                        for n in graph_nodes:
                            st.write(
                                f"• **{n['node_type']}** — "
                                f"`{n['top_feature']}` — attribution {n['attribution_score']:.5f}"
                            )
                        st.caption("Attribution = |gradient × input|. Model evidence, not causal proof.")

            st.divider()

            # Local graph visualization
            st.markdown("**Local graph (conceptual visualization)**")
            st.caption(
                "Based on the values you entered. "
                "Dashed red edges indicate potential ring connections "
                "when the device is shared across many accounts."
            )
            try:
                fig = _draw_local_graph(
                    amount, avg_amt, hour, xcb, merch_r,
                    int(dev_accts), dev_risk, velocity, balance,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.info(f"Graph unavailable: {e}")

            caveats = result.get("caveats", [])
            if caveats:
                with st.expander("Caveats", expanded=False):
                    for c in caveats:
                        st.caption(f"⚠ {c}")
                    st.caption(result.get("warning", ""))

            with st.expander("🔬 Teaching experiments — what to try next", expanded=False):
                st.markdown(f"""
**Your current setup** — Amount Z-score: {zscore:+.2f}, Device accounts: {int(dev_accts)}, Device risk: {dev_risk:.2f}

| Experiment | Change to make | What to observe |
|------------|---------------|----------------|
| Isolate ring signal | Device accounts = 10, device risk = 0.85; keep A-section values normal | Does score rise from graph context alone? |
| Saturate merchant | Merchant risk = 0.90 | Does merchant risk alone shift the decision? |
| Velocity burst | Velocity = 15+ | Burst signal vs. normal |
| Normalize amount | Account average = same as amount | Z-score → 0; does risk drop? |
| Unusual hour + cross-border | Hour = 2, check cross-border | Combined vs. individual effect |
| Compare presets | Use preset selector above | Normal vs. Ring vs. ATO side by side |

> The model sees all features simultaneously — changing one signal may not produce a dramatic shift.
                """)

    else:
        if model_ready:
            st.info(
                "**Choose a preset above, then click 🔍 Analyze Transaction.**  \n"
                "Start with **Suspicious Device Ring** — it demonstrates the GNN's core "
                "advantage over tabular models."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Teaching Scenarios
# ─────────────────────────────────────────────────────────────────────────────
with tab_scenarios:
    st.subheader("Teaching scenarios")
    st.markdown(
        "These are **pre-defined teaching examples**, not cherry-picked results. "
        "They illustrate specific fraud patterns to show what the model can and "
        "cannot detect. Scores are **actual model outputs**, not fabricated values."
    )

    # Scenario definitions
    SCENARIOS = {
        "Normal transaction": {
            "desc": (
                "A routine domestic transfer within the account's normal range. "
                "Normal hour, private device, low-risk merchant. "
                "**Expected outcome:** Both tabular and GNN models assign low risk."
            ),
            "signal": "All features and context are benign.",
            "amount": 450.0, "avg_amt": 400.0, "hour": 14, "xcb": False,
            "merch_r": 0.05, "balance": 8000.0, "velocity": 2.0,
            "acct_age": 365.0, "cust_age": 35, "tenure": 730, "c_risk": 1,
            "dev_accts": 1, "dev_cust": 1, "dev_risk": 0.04,
        },
        "Ring fraud (suspicious neighbourhood)": {
            "desc": (
                "Transaction amount and hour look **completely normal**. "
                "BUT the account's device is shared with 10 other accounts, "
                "and the device risk score is high (planted ring pattern). "
                "**Expected outcome:** Tabular baseline misses this; "
                "GNN may assign higher risk due to neighbourhood context. "
                "*(Score depends on trained model — actual output shown.)*"
            ),
            "signal": "Normal transaction features; suspicious shared-device neighbourhood.",
            "amount": 380.0, "avg_amt": 350.0, "hour": 13, "xcb": False,
            "merch_r": 0.06, "balance": 4500.0, "velocity": 3.0,
            "acct_age": 200.0, "cust_age": 28, "tenure": 400, "c_risk": 1,
            "dev_accts": 10, "dev_cust": 8, "dev_risk": 0.82,
        },
        "Suspicious features, moderate context": {
            "desc": (
                "High amount ($12,000 vs average $900), unusual hour (02:30), "
                "cross-border. **BUT** private device (1 account), low-risk merchant, "
                "long-tenured customer. "
                "**Expected outcome:** Tabular baseline flags this strongly; "
                "GNN considers context — actual score shown."
            ),
            "signal": "Suspicious transaction features; moderately clean graph context.",
            "amount": 12000.0, "avg_amt": 900.0, "hour": 2, "xcb": True,
            "merch_r": 0.05, "balance": 50000.0, "velocity": 1.5,
            "acct_age": 1200.0, "cust_age": 52, "tenure": 3650, "c_risk": 1,
            "dev_accts": 1, "dev_cust": 1, "dev_risk": 0.03,
        },
        "Raushan transfers $2,000": {
            "desc": (
                "Raushan (age 32, 4-year tenure) transfers $2,000. Account average "
                "is $1,500, so the amount is within normal range. Domestic, "
                "normal hour, private device. "
                "**Expected outcome:** Low risk from both models."
            ),
            "signal": "All features normal — reference baseline transaction.",
            "amount": 2000.0, "avg_amt": 1500.0, "hour": 15, "xcb": False,
            "merch_r": 0.04, "balance": 12000.0, "velocity": 2.0,
            "acct_age": 480.0, "cust_age": 32, "tenure": 1460, "c_risk": 1,
            "dev_accts": 1, "dev_cust": 1, "dev_risk": 0.03,
        },
    }

    # Initialize session state for scenario results
    if "scenario_results" not in st.session_state:
        st.session_state.scenario_results = {}

    # "Run all scenarios" button
    if st.button("▶ Run all 4 scenarios", disabled=not model_ready, type="primary"):
        for name, sc in SCENARIOS.items():
            pl = _build_payload(
                sc["amount"], sc["avg_amt"], sc["hour"], sc["xcb"], sc["merch_r"],
                sc["balance"], sc["velocity"], sc["acct_age"], sc["cust_age"],
                sc["tenure"], sc["c_risk"], sc["dev_accts"], sc["dev_cust"],
                sc["dev_risk"],
            )
            with st.spinner(f"Scoring: {name}…"):
                st.session_state.scenario_results[name] = _call_score(pl)

    st.divider()

    # Display scenarios in a 2×2 grid
    sc_names = list(SCENARIOS.keys())
    row1 = st.columns(2)
    row2 = st.columns(2)
    pairs = [(row1[0], sc_names[0]), (row1[1], sc_names[1]),
             (row2[0], sc_names[2]), (row2[1], sc_names[3])]

    for col, name in pairs:
        with col:
            sc = SCENARIOS[name]
            st.markdown(f"### {name}")
            st.caption(sc["signal"])
            st.markdown(sc["desc"])

            # Individual run button
            if st.button(f"Score: {name[:20]}…", key=f"sc_{name}", disabled=not model_ready):
                pl = _build_payload(
                    sc["amount"], sc["avg_amt"], sc["hour"], sc["xcb"], sc["merch_r"],
                    sc["balance"], sc["velocity"], sc["acct_age"], sc["cust_age"],
                    sc["tenure"], sc["c_risk"], sc["dev_accts"], sc["dev_cust"],
                    sc["dev_risk"],
                )
                st.session_state.scenario_results[name] = _call_score(pl)

            if name in st.session_state.scenario_results:
                r = st.session_state.scenario_results[name]
                _render_score_result(r, name)

                # Show mini graph
                try:
                    fig = _draw_local_graph(
                        sc["amount"], sc["avg_amt"], sc["hour"],
                        sc["xcb"], sc["merch_r"],
                        sc["dev_accts"], sc["dev_risk"],
                        sc["velocity"], sc["balance"],
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    pass
            st.divider()

    # Side-by-side comparison table (if all 4 ran)
    if len(st.session_state.scenario_results) == 4:
        st.subheader("Side-by-side comparison")
        st.caption(
            "All scores are actual GNN outputs. The key question: does the ring-fraud "
            "scenario score higher than the normal scenario even though its transaction "
            "features look similar?"
        )
        rows = []
        for name, sc in SCENARIOS.items():
            r = st.session_state.scenario_results.get(name, {})
            score = r.get("fraud_risk_score", "—")
            dec   = r.get("decision", "—")
            rows.append({
                "Scenario":    name,
                "Amount":      f"${sc['amount']:,.0f}",
                "Device accts": sc["dev_accts"],
                "Fraud Risk Score": f"{score:.4f}" if isinstance(score, float) else score,
                "Decision":    dec,
            })
        import pandas as pd
        st.table(pd.DataFrame(rows).set_index("Scenario"))
        st.caption(
            "⚠ Teaching examples — scores from a model trained on synthetic data. "
            "Do not interpret as real fraud predictions."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Graph Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_graph:
    st.subheader("Graph schema and dataset")

    # Fetch stats
    try:
        sample = requests.get(f"{BACKEND}/gnn/graph-sample", timeout=5).json()
    except Exception:
        sample = {}

    if sample.get("key_educational_point"):
        st.info(sample["key_educational_point"])

    g_left, g_right = st.columns([1, 1])

    with g_left:
        st.markdown("### Why a graph instead of a table?")
        st.markdown(
            """
In a flat transaction table, every row describes one transaction in isolation.
You cannot express the relationship:

> *"Account A shares a device with Account B, C, and D — all of which have
> fraudulent transaction histories."*

That relationship only exists as **edges** connecting nodes in a graph.
A GNN propagates signals along those edges, so the device's fraud history
can reach the transaction through two message-passing hops:

```
Transaction ← account ← device ← peer accounts with fraud
```

### Node types

| Icon | Node | What it represents | Features |
|------|------|--------------------|---------|
| 🧑 | Customer | Account holder | age, tenure, country risk, #accts |
| 🏦 | Account | Bank account | balance, avg amount, velocity, age |
| 💳 | Transaction | Payment **(fraud label)** | amount, z-score, hour, cross-border |
| 🏪 | Merchant | Payment recipient | risk score, #txns, country risk |
| 📱 | Device | Device fingerprint | #accts sharing, #customers, risk |

### Edge types

| Edge | Direction | Why it exists |
|------|-----------|---------------|
| `owns` | Customer → Account | Identity context flows to account |
| `makes` | Account → Transaction | Account history informs transaction risk |
| `paid_to` | Transaction → Merchant | Merchant fraud history propagates back |
| `uses` (cust) | Customer → Device | Device sharing at customer level |
| `uses` (acct) | Account → Device | Device sharing at account level (stronger signal) |

Each forward edge has a **reverse edge** so signals can flow in both directions
during message passing.
            """
        )

    with g_right:
        st.markdown("### Ring fraud diagram")
        try:
            fig = _draw_ring_fraud_diagram()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"Diagram unavailable: {e}")

        st.markdown(
            """
**How the GNN detects the ring:**

1. **Layer 1** — Account A receives a message from Device X.
   Device X's features include `num_accounts=11` and `device_risk=0.85`.

2. **Layer 2** — Transaction A's embedding now contains information that
   passed through Account A, which in turn received it from Device X.

3. The MLP head scores Transaction A using this enriched embedding —
   which now "knows" that its account shares a suspicious device.

A tabular model that only sees Transaction A's own row cannot reach this signal.
            """
        )

    # Dataset statistics from backend
    if sample and "node_types" in sample:
        st.divider()
        st.subheader("Synthetic dataset statistics")
        st.caption("⚠ All numbers describe generated data, not real transactions.")

        cols = st.columns(5)
        for i, (nt, info) in enumerate(sample["node_types"].items()):
            with cols[i]:
                icon_map = {"customer": "🧑", "account": "🏦", "transaction": "💳",
                            "merchant": "🏪", "device": "📱"}
                st.metric(
                    f"{icon_map.get(nt,'')} {nt.capitalize()}",
                    f"{info['approx_count']:,}",
                )
                st.caption(info["description"])

        st.divider()
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Fraud patterns in the dataset**")
            for pattern, desc in sample.get("fraud_patterns", {}).items():
                badge = "🔴" if "ring" in pattern.lower() else "🟠"
                st.write(f"{badge} **{pattern}**: {desc}")
            st.metric("Overall fraud rate", sample.get("fraud_rate_pct", "~5.8%"))

        with pcol2:
            st.markdown("**Why four different fraud patterns?**")
            st.markdown(
                """
The dataset deliberately mixes two broad types:

| Type | Patterns | Tabular detectable? |
|------|----------|---------------------|
| **Tabular fraud** | A (amount+hour+cross-border) | ✅ Yes |
| **Relational fraud** | B (ring), C (merchant), D (velocity) | ❌ Mostly no |

~84% of fraud is relational. This means a tabular model will miss the
majority of planted fraud, while the GNN — which sees graph relationships —
should perform better on the relational subset.
            """
            )

        st.caption(sample.get("split", "Split: temporal 70/15/15"))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Why GNN?
# ─────────────────────────────────────────────────────────────────────────────
with tab_why:
    st.subheader("Why use a Graph Neural Network for fraud detection?")

    why_left, why_right = st.columns([1, 1])

    with why_left:
        st.markdown("### Traditional ML approach")
        st.markdown(
            """
```
┌─────────────────────────────────────────────────┐
│  One transaction row                            │
│  ─────────────────────────────────────────────  │
│  amount        =  $450                          │
│  hour          =  14:00                         │
│  cross_border  =  False                         │
│  merchant_risk =  0.06                          │
│  amount_zscore =  +0.3  (near average)          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Logistic Reg.  │
         │    LightGBM     │
         └────────┬────────┘
                  │
                  ▼
         Risk Score = 0.08  →  APPROVE
```

The model asks:
> *"Does this single row look suspicious?"*

**What it cannot see:**
- Which device the account uses
- Whether that device is shared with flagged accounts
- The merchant's historical fraud rate across all transactions
- Whether this account is part of a coordinated group
            """
        )

    with why_right:
        st.markdown("### GNN approach")
        st.markdown(
            """
```
┌──────────┐   owns   ┌──────────┐   makes  ┌────────────┐
│ Customer │ ───────► │ Account  │ ─────────►│Transaction │
│ age=28   │          │ bal=4500 │           │ $450       │
│ tenure=  │          │ vel=3.0  │           │ hour=13    │
│  400d    │          └────┬─────┘           └─────┬──────┘
└──────────┘               │                       │
                          uses                  paid_to
                           │                       │
                     ┌─────▼─────┐           ┌────▼────┐
                     │  Device X │           │Merchant │
                     │ 10 accts  │           │risk=0.06│
                     │ risk=0.82 │           └─────────┘
                     └─────┬─────┘
                           │ (ring connections)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Acct B         Acct C       Acct D
        (fraud)        (fraud)      (fraud)
```

```
     Message passing (2 layers)
           │
           ▼
     Transaction embedding:
     now "knows" about Device X
     and its fraudulent connections
           │
           ▼
     Risk Score = ?  ← actual model output
```

The model asks:
> *"Does this transaction look suspicious given its own features**
> AND what it is connected to?"*
            """
        )

    st.divider()
    st.subheader("The key difference in plain language")

    kc1, kc2 = st.columns(2)
    with kc1:
        st.info(
            "**Traditional ML:**  \n"
            "\"This transaction's amount, hour, and merchant look normal.\"  \n"
            "→ Score based on the transaction row alone."
        )
    with kc2:
        st.warning(
            "**GNN:**  \n"
            "\"This transaction looks normal, BUT its account shares a device "
            "with 8 other accounts that all have fraud histories.\"  \n"
            "→ Score incorporates graph-neighbourhood evidence."
        )

    st.divider()
    st.subheader("When does graph information actually help?")
    st.markdown(
        """
| Fraud type | Tabular baseline | GNN | Why |
|------------|-----------------|-----|-----|
| High amount + unusual hour | ✅ Detects | ✅ Detects | Transaction features are suspicious |
| **Shared-device ring** | ❌ Misses | ✅ Detects | Features look normal; only graph reveals ring |
| High-risk merchant (static) | Partial | Better | Merchant feature available in tabular too, but GNN aggregates richer context |
| Velocity burst | Partial | Partial | Time-since-last-txn partially captures this |

The **relational fraud patterns** (especially shared-device rings) are where graph
structure provides the clearest advantage.

The notebook (`04_gnn_fraud_detection.ipynb`) measures this empirically by
reporting recall specifically on Pattern B (ring fraud) for both the tabular
baseline and the GNN — the column that answers "what does the graph add?"
        """
    )

    st.divider()
    st.subheader("Message passing — how does the GNN 'see' the neighbourhood?")
    st.markdown(
        """
Imagine each node holding a notecard with its features written on it.

**Before message passing:** each node only knows its own features.

**After Layer 1:** each node gets a new notecard that blends:
- its own features
- the average of its direct neighbours' features

So after Layer 1:
- The **Account** node now has a mixed notecard containing account features
  + device features + customer features.
- The **Transaction** node now has a mixed notecard containing transaction
  features + account features + merchant features.

**After Layer 2:** each node blends again with its (already updated) neighbours.

So after Layer 2:
- The **Transaction** node's notecard now contains (indirectly):
  transaction → account → device information.
  It can now "see" whether the device has many connected accounts.

This is called **2-hop message passing** — it reaches 2 steps away from the
target node, which is exactly how far the ring-device signal needs to travel.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — How It Works
# ─────────────────────────────────────────────────────────────────────────────
with tab_how:
    st.subheader("Architecture and implementation")

    hw_left, hw_right = st.columns([1, 1])

    with hw_left:
        st.markdown("### Model architecture")
        st.markdown(
            """
```
Input features (per node type — different dimensions)
  customer:    4 features
  account:     4 features
  transaction: 8 features  ← fraud labels here
  merchant:    3 features
  device:      3 features
        │
        ▼
Linear projection  →  64-dim shared embedding (per node type)
        │
        ▼  ─────── HeteroConv Layer 1 ────────
       SAGEConv(64→64) per edge type
       + Residual connection
       + LayerNorm
       + ReLU
       + Dropout(0.3)
        │
        ▼  ─────── HeteroConv Layer 2 ────────
       (same structure)
        │
        ▼
Transaction node embeddings  [N × 64]
        │
        ▼  ─── MLP Classification Head ───
       Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1)
        │
        ▼
Raw logit  →  sigmoid  →  Fraud Risk Score ∈ [0, 1]
```
            """
        )

        st.markdown("### Why SAGEConv?")
        st.markdown(
            """
SAGEConv (Hamilton et al., 2017) computes:

```
h_v = W · concat(h_v,  mean(h_u for u in neighbours(v)))
```

For our transaction graph — where each transaction touches exactly one account,
one merchant, and (via the account) one device — the neighbourhood is sparse
and well-defined.  Mean aggregation handles this cleanly.

Attention-based variants (GAT) compute different weights for different
neighbours, which is useful when a node has many diverse neighbours.
SAGEConv's simpler mean aggregation is better suited here.
            """
        )

    with hw_right:
        st.markdown("### Trace one transaction end-to-end")
        st.markdown(
            """
Let's follow a ring-fraud transaction through every step:

**Step 1 — Raw features**
```
Transaction T:
  amount       = $380
  amount_zscore = +0.09   (nearly at average — looks normal)
  hour         = 13       (normal business hours)
  cross_border = False
  merchant_risk = 0.06    (safe merchant)

Account A:
  balance   = $4,500
  velocity  = 3.0 txns/day
  age       = 200 days

Device X:
  num_accounts = 10        ← KEY SIGNAL
  device_risk  = 0.82      ← KEY SIGNAL
```

**Step 2 — Linear projection**
Each node's raw features are projected into a shared 64-dim space.

**Step 3 — Layer 1 message passing**
```
Account A receives:
  → its own features (balance, velocity…)
  → mean of Device X's features (num_accounts=10, risk=0.82)
  → Customer's features (age, tenure…)

Transaction T receives:
  → its own features (amount, hour…)
  → Account A's (now updated) features
  → Merchant's features
```

**Step 4 — Layer 2 message passing**
```
Transaction T receives:
  → Account A's embedding (which now "knows" about Device X)
  → Merchant embedding

Transaction T now "knows" (indirectly):
  that its account uses a device shared with 9 other accounts
  and that device has a risk score of 0.82
```

**Step 5 — MLP head**
Transaction T's 64-dim embedding is passed through the classifier:
```
Linear(64→32) → ReLU → Dropout → Linear(32→1) → sigmoid
```

**Step 6 — Score and decision**
```
Fraud Risk Score = 0.xx   (actual model output)
→ APPROVE / REVIEW / BLOCK  (based on configurable thresholds)
```
            """
        )

    st.divider()
    st.subheader("Explainability: gradient × input attribution")
    st.markdown(
        """
After scoring, the explanation is computed by:

1. Running a forward pass to get the fraud risk logit for this transaction.
2. Backpropagating the logit through the full graph back to the input features.
3. Computing `|gradient × input|` for each feature of each node type.
4. Ranking node types and features by their attribution score.

**What this means:**
Attribution scores show which input features the model gave most weight to
when computing this specific score.  A high attribution for `device.num_accounts`
means the model's score was sensitive to how many accounts share this device.

**What this does NOT mean:**
- It does not prove that device sharing *caused* the fraud in the real world.
- It does not tell you the exact fraction of the score "caused" by each feature.
- Attribution can vary between runs and is an approximation.
- The explanation reflects the synthetic training patterns — not real fraud causality.

All explanations shown in this UI are accompanied by this caveat.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Model & Limitations
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.subheader("Model information and limitations")

    info_col, limit_col = st.columns([1, 1])

    with info_col:
        st.markdown("### Model status")
        if model_ready:
            st.success("GNN model is loaded and serving requests.")
        else:
            st.error("GNN model is NOT loaded.")
            st.markdown(
                "**To train and load the model:**\n\n"
                "1. Open `notebooks/04_gnn_fraud_detection.ipynb`\n"
                "2. Run all cells (takes ~3–5 minutes on CPU)\n"
                "3. The notebook saves `models/gnn_fraud_model.pt`\n"
                "4. Restart the backend: `uvicorn main:app --app-dir backend --reload --port 8000`\n"
                "5. Refresh this page — the banner above will turn green."
            )

        st.markdown("### Architecture")
        st.markdown(
            """
| Property | Value |
|----------|-------|
| Model type | Heterogeneous GNN (HeteroConv + SAGEConv) |
| Message-passing layers | 2 |
| Embedding dimension | 64 |
| Dropout | 0.3 |
| Classification head | Linear(64→32)→ReLU→Linear(32→1) |
| Parameters | ~170,241 |
| Loss | BCEWithLogitsLoss + pos_weight |
| Optimiser | Adam, lr=5e-3, wd=1e-4 |
| Early stopping | Val PR-AUC, patience=10 |
            """
        )

        st.markdown("### Graph")
        st.markdown(
            """
| Property | Value |
|----------|-------|
| Node types | customer, account, transaction, merchant, device |
| Edge types | 5 forward + 5 reverse = 10 total |
| Transactions | ~56,000 |
| Fraud rate | ~5.8% |
| Fraud patterns | A (tabular), B (ring), C (merchant), D (velocity) |
| Split | Temporal 70/15/15 |
            """
        )

        st.markdown("### API endpoints")
        st.markdown(
            """
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/gnn/status` | GET | Model health check |
| `/gnn/score` | POST | Score a transaction |
| `/gnn/demo/{1-3}` | GET | Pre-defined demos |
| `/gnn/graph-sample` | GET | Dataset statistics |
            """
        )

    with limit_col:
        st.markdown("### Limitations")
        st.warning("These limitations are important to understand before interpreting any score.")
        st.markdown(
            """
**1. Synthetic data only**
The model learned patterns that were deliberately planted. It proves that
a GNN *can* use graph information — not that it will generalise to real fraud.

**2. Graph leakage in the train/test split**
Accounts, devices, and merchants appear in both train and test sets.
The GNN has seen their graph structure during training, which inflates
test metrics compared to a truly cold-start deployment.

**3. Single-transaction API inference does not detect rings**
The `/gnn/score` endpoint builds a minimal 5-node subgraph. It cannot
detect shared-device patterns from the full training graph. Device risk
is inferred only from the values you enter, not from actual graph connections.
Full ring detection requires the pre-built training graph.

**4. Explainability is approximate**
Gradient × input attribution is a fast approximation. It shows what the
model weighted, not what caused fraud. Results may vary slightly.

**5. No model calibration**
The output is a "Fraud Risk Score," not a calibrated fraud probability.
A score of 0.70 does not mean a 70% chance of real fraud.

**6. Decision thresholds are illustrative**
The APPROVE/REVIEW/BLOCK thresholds (0.30 and 0.70 by default) are
configurable examples — not universal fraud-detection policy.

**7. Real-world deployment would require**
- Real labelled transaction data
- Proper entity-disjoint train/test splits
- Model calibration
- Regulatory review
- Privacy controls for device fingerprint data
            """
        )

        st.markdown("### Files in this module")
        st.markdown(
            """
```
src/camel_sentinel/gnn/
  synthetic_data.py   — data generation
  graph_builder.py    — DataFrames → PyG HeteroData
  model.py            — HeteroGNN architecture
  trainer.py          — baselines + GNN training
  explainer.py        — gradient attribution
  detector.py         — inference + policy

backend/gnn_router.py — FastAPI routes
notebooks/04_gnn_fraud_detection.ipynb — full teaching notebook
docs/05_gnn_fraud_detection.md — design reference
```
            """
        )

        st.markdown("### Citation / methodology")
        st.markdown(
            """
- **SAGEConv**: Hamilton, W., Ying, Z., & Leskovec, J. (2017).
  *Inductive Representation Learning on Large Graphs.*
  NeurIPS 2017.
- **PyTorch Geometric**: Fey, M. & Lenssen, J.E. (2019).
  *Fast Graph Representation Learning with PyTorch Geometric.*
  ICLR 2019 Workshop.
- **GNNExplainer**: Ying, R. et al. (2019).
  *GNNExplainer: Generating Explanations for Graph Neural Networks.*
  NeurIPS 2019.
            """
        )
