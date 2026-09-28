"""
frontend/app.py — the app's entry point and navigation.

`streamlit run frontend/app.py` starts here. This file only:
  1. groups the pages into sidebar sections,
  2. applies the shared styling,
  3. renders the credits (sidebar) and footer on EVERY page,
then runs the selected page. The scoring UI that used to live here is now
frontend/home.py. Pages still talk to the backend over HTTP only.

Developer: Raushan Ranjan (https://raushan-ranjan.azurewebsites.net) · Books/Handbook/Notes written by Raushan Ranjan.
"""
from __future__ import annotations

import streamlit as st

from ui_kit import apply_global_style, backend_panel, footer, sidebar_credits, theme_selector

P = "views/"  # not "pages/": Streamlit would auto-discover that folder and clash with st.navigation
nav = st.navigation({
    "CAMEL Sentinel": [
        st.Page("home.py", title="Score a Bank", icon="🐫", default=True),
        st.Page(P + "2_💬_Chat_Assistant.py", title="Chat Assistant", icon="💬", url_path="chat"),
        st.Page(P + "1_📚_Documentation.py", title="Documentation", icon="📚", url_path="documentation"),
    ],
    "Labs": [
        st.Page(P + "3_🕸️_Fraud_Detection.py", title="Fraud Detection (GNN)", icon="🕸️", url_path="fraud-detection"),
        st.Page(P + "6_🛡️_AI_Security_Lab.py", title="AI Security Lab", icon="🛡️", url_path="security-lab"),
    ],
    "Learn & Ship": [
        st.Page(P + "7_🗺️_AI_ML_Roadmap.py", title="AI/ML Roadmap", icon="🗺️", url_path="roadmap"),
        st.Page(P + "4_🎓_Build_Guide.py", title="Build Guide", icon="🎓", url_path="build-guide"),
        st.Page(P + "5_🚢_Deploy_and_Containerize.py", title="Deploy & Containerize", icon="🚢", url_path="deploy"),
    ],
    "Trust": [
        st.Page(P + "8_🔐_Security_Posture.py", title="Security Posture", icon="🔐", url_path="security-posture"),
    ],
})

theme_selector()
backend_panel()
apply_global_style()
sidebar_credits()
nav.run()
footer()
