"""
frontend/views/5_🚢_Deploy_and_Containerize.py

Learning path stage: MLOps & Deployment. Explains, demonstrates and gives
step-by-step instructions for packaging CAMEL Sentinel as a Docker image,
publishing it to Docker Hub and Azure Container Registry, and running it in
the cloud. Design + runbook: docs/06_deployment_containerization.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_kit import PROJECT_ROOT, backend_url, handbook_credit, html, in_container, learn_link  # noqa: E402

st.set_page_config(page_title="Deploy & Containerize · CAMEL Sentinel", page_icon="🚢", layout="wide")
BACKEND_URL = backend_url()

DOCKERHUB_REPO = "raushanranjan/camel-sentinel"
ACR_SERVER = "camelsentinelacr.azurecr.io"
ACR_REPO = f"{ACR_SERVER}/camel-sentinel"
RG = "rg-camel-sentinel"


def read_file(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else f"# {rel} not found in this copy of the project"


# Real `docker history camel-sentinel:v1.0` output (captured when v1.0 was built).
V1_LAYERS = [
    ("Debian base filesystem (python:3.10-slim)", 87.6, "base"),
    ("Debian: ca-certificates, tzdata", 4.95, "base"),
    ("Python 3.10.21 interpreter", 46.0, "base"),
    ("apt: libgomp1, libasound2, libssl3, curl", 15.6, "system"),
    ("requirements-docker.txt", 0.01, "deps"),
    ("pip: torch (CPU) + torch-geometric + 20 packages", 1990.0, "deps"),
    ("src/ (camel_sentinel package)", 0.32, "code"),
    ("backend/ (FastAPI)", 0.05, "code"),
    ("frontend/ (Streamlit)", 0.33, "code"),
    ("models/ (.joblib, .pt, .pkl)", 3.01, "assets"),
    ("data/ (panel + GNN parquet)", 19.7, "assets"),
    ("notebooks/ result CSVs + docs/", 0.25, "assets"),
    ("useradd + chown -R /app  ← duplicated files!", 23.7, "waste"),
]

# ═════════════════════════════════════════════════════════════════════════════
st.title("🚢 Deploy & Containerize")
st.caption("Learning path · **10 · MLOps & Deployment** — from “works on my laptop” to “runs anywhere”.")
handbook_credit()

c1, c2, c3 = st.columns(3)
with c1:
    if in_container():
        st.success("🐳 You are viewing this page **inside a Docker container** right now.")
    else:
        st.info("💻 Running directly on this machine (not in a container).")
with c2:
    try:
        ok = requests.get(f"{BACKEND_URL}/health", timeout=3).json().get("model_loaded")
        st.success(f"API healthy at `{BACKEND_URL}`" if ok else "API up, model loading…")
    except requests.RequestException:
        st.warning(f"API not reachable at `{BACKEND_URL}`")
with c3:
    st.markdown(f"**Published images**  \n[Docker Hub ↗](https://hub.docker.com/r/{DOCKERHUB_REPO}) · `{ACR_SERVER}`")

tabs = st.tabs(["🧭 Big picture", "📄 Dockerfile explained", "🛠️ Do it manually", "🤖 Do it with AI",
                "📦 Registries", "☁️ Deploy to cloud", "🔐 DevSecOps", "⚙️ Behind the scenes", "🧯 Troubleshooting"])

# ── 1. Big picture ────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("The journey of one image")
    st.markdown("Click any stage to see what happens there. The dot shows the image travelling from your code to users.")
    stages = [
        ("💻", "Code", "Your project folder", "backend/, frontend/, src/, models/, data/. On your laptop it only runs because Python, torch, sklearn 1.7.2… happen to be installed exactly right. That is the “works on my machine” problem."),
        ("📄", "Dockerfile", "The recipe", "A text file of steps: start from python:3.10-slim, install system libs, pip install pinned requirements, copy code, switch to a non-root user, declare a health check and a start command."),
        ("🧱", "Image", "docker build", "Docker runs each step and freezes the result as a read-only layer. Layers stack into an image (~650 MB compressed for CAMEL Sentinel). Same image = same behaviour everywhere."),
        ("📦", "Registry", "docker push", "Upload the image to a registry: Docker Hub (public, raushanranjan/camel-sentinel) and Azure Container Registry (private, camelsentinelacr.azurecr.io). Only layers the registry doesn't have are uploaded."),
        ("☁️", "Cloud", "docker run / Container Apps", "Any machine with a container runtime pulls the layers and starts a container: Azure Container Apps, Kubernetes (AKS), App Service, or a colleague's laptop. Secrets are injected here, at run time."),
        ("👩‍💼", "Users", "https://…", "The analyst opens the URL. Streamlit (8501) serves the UI and calls FastAPI (8000) inside the same container, which serves the model, SHAP, the Security Lab…"),
    ]
    items = "".join(
        f'<div class="st" data-i="{i}" onclick="pick({i})"><div class="ic">{ic}</div><b>{t}</b><span class="muted">{s}</span></div>'
        + ('<div class="ar">➜</div>' if i < len(stages) - 1 else "")
        for i, (ic, t, s, _) in enumerate(stages)
    )
    details = json.dumps([d for *_, d in stages])
    html(f"""
<style>
.flow{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;justify-content:center;position:relative;padding:8px 0}}
.st{{flex:1 1 120px;max-width:160px;min-width:110px;text-align:center;cursor:pointer;background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:12px 8px;display:flex;flex-direction:column;gap:2px;transition:transform .2s,border-color .2s,box-shadow .2s}}
.st:hover{{transform:translateY(-3px)}} .st.on{{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent)}}
.st span{{font-size:12px}} .ic{{font-size:28px}}
.ar{{color:var(--muted);font-size:20px;animation:pulse 1.6s infinite}}
@keyframes pulse{{0%,100%{{opacity:.35}}50%{{opacity:1}}}}
#d{{margin-top:12px;min-height:64px}}
.bar{{height:6px;border-radius:6px;background:var(--line);overflow:hidden;margin-top:10px}}
.bar i{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--teal));animation:go 6s linear infinite}}
@keyframes go{{from{{width:0}}to{{width:100%}}}}
@media (max-width:640px){{.ar{{display:none}}}}
</style>
<div class="flow">{items}</div>
<div class="bar"><i></i></div>
<div id="d" class="card"></div>
<script>
const D={details};
function pick(i){{document.querySelectorAll('.st').forEach(e=>e.classList.toggle('on',+e.dataset.i===i));
document.getElementById('d').innerHTML='<b>Step '+(i+1)+'</b> — '+D[i];}}
let k=0;pick(0);const t=setInterval(()=>{{k=(k+1)%D.length;pick(k)}},3500);
document.querySelectorAll('.st').forEach(e=>e.addEventListener('click',()=>clearInterval(t)));
</script>""", height=290)

    st.subheader("Image vs container — the one distinction to remember")
    a, b = st.columns(2)
    a.markdown("""
**Image** 🧊 — *the frozen meal*
- Read-only, built once, versioned by **tag** (`v1.0`) and **digest** (`sha256:…`)
- Stored in a registry, shared by everyone
- Contains code + dependencies + models, **never secrets**
""")
    b.markdown("""
**Container** 🍽️ — *the meal heated and served*
- A running process started *from* an image
- Gets its own filesystem layer, network, and process space
- Many containers can run from one image (3 replicas on Kubernetes)
""")
    st.subheader("One image, three ways to run it")
    st.markdown("The same image starts different processes depending on the `MODE` environment variable (`docker/docker-entrypoint.sh`).")
    st.table([
        {"MODE": "all (default)", "Starts": "API :8000 in background + UI :8501", "Use for": "Single-container cloud (Azure Container Apps), class demos"},
        {"MODE": "api", "Starts": "FastAPI only", "Use for": "Scaling the API separately, Kubernetes"},
        {"MODE": "ui", "Starts": "Streamlit only, calling CAMEL_BACKEND_URL", "Use for": "docker-compose two-service setup"},
    ])
    learn_link(10, "FastAPI serving & Docker")

# ── 2. Dockerfile explained ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("The real Dockerfile of this project, line by line")
    left, right = st.columns([1.1, 1])
    with left:
        st.code(read_file("Dockerfile"), language="docker")
    with right:
        explain = {
            "FROM python:3.10-slim": "Start from an official, minimal Debian + Python 3.10 image (~120 MB). Same Python version the models were trained with.",
            "ENV …": "Settings baked into every container: unbuffered logs so `docker logs` is live, no pip cache, Streamlit headless without telemetry, default MODE=all.",
            "RUN apt-get …": "OS libraries Python wheels need: libgomp1 (LightGBM/sklearn threads), libasound2 + libssl3 (Azure Speech SDK), curl (health check). Clean apt lists in the *same* layer, or the layer keeps them forever.",
            "WORKDIR /app": "All following paths are relative to /app.",
            "COPY requirements-docker.txt + RUN pip …": "Dependencies BEFORE code. Docker caches this 2 GB layer; editing a .py file later rebuilds in seconds, not 10 minutes. torch==2.13.0+cpu avoids ~2.5 GB of unused CUDA.",
            "RUN useradd …": "Create an unprivileged user `camel` (uid 10001). Created before copying code so files can be copied already owned by it.",
            "COPY --chown=camel:camel …": "Copy models, data, then code, owned by camel. Least-changing things first, most-changing last, for the best cache reuse.",
            "USER camel": "Everything from here, including the app, runs as a non-root user. A compromised app is not root.",
            "EXPOSE 8000 8501": "Documentation: which ports the container listens on. You still publish them with `-p` at run time.",
            "HEALTHCHECK …": "Docker (and every orchestrator) runs docker/healthcheck.sh every 15 s: healthy only when /health reports model_loaded=true (or, in ui mode, Streamlit answers). `start-period=120s` covers model + SHAP + LIME warm-up.",
            "ENTRYPOINT […]": "The start command: docker-entrypoint.sh reads MODE and starts the API, the UI, or both.",
        }
        pick = st.radio("Pick an instruction", list(explain), label_visibility="collapsed")
        st.info(explain[pick])
        st.markdown("**What's *not* in the image** (`.dockerignore`): `.env` secrets, `Project-versions/`, zip archives, virtualenvs, caches, notebooks' heavy outputs.")
        with st.expander(".dockerignore"):
            st.code(read_file(".dockerignore"), language="bash")

    st.subheader("🧪 Layer cache simulator")
    st.markdown("What did you change? See which layers Docker must rebuild (red) and which it reuses from cache (green).")
    change = st.selectbox("Change", ["Edited a Python file in frontend/", "Retrained a model in models/",
                                     "Added a package to requirements-docker.txt", "Changed the FROM base image"])
    order = ["FROM python:3.10-slim", "apt-get install", "COPY requirements", "pip install (2 GB)", "useradd",
             "COPY models", "COPY data", "COPY src/backend", "COPY frontend"]
    first_dirty = {"Edited a Python file in frontend/": 8, "Retrained a model in models/": 5,
                   "Added a package to requirements-docker.txt": 2, "Changed the FROM base image": 0}[change]
    secs = [0, 20, 0, 540, 1, 1, 1, 1, 1]
    rebuild = sum(secs[first_dirty:])
    rows = "".join(
        f'<div class="ly {"dirty" if i >= first_dirty else "cached"}" style="animation-delay:{i*0.08}s">'
        f'<span>{name}</span><span>{"🔨 rebuild" if i >= first_dirty else "✅ cached"}</span></div>'
        for i, name in enumerate(order))
    html(f"""
<style>.ly{{display:flex;justify-content:space-between;padding:7px 12px;margin:3px 0;border-radius:8px;
opacity:0;animation:in .4s forwards;font-size:13px}}
.cached{{background:color-mix(in srgb,var(--ok) 18%,transparent);border:1px solid color-mix(in srgb,var(--ok) 40%,transparent)}}
.dirty{{background:color-mix(in srgb,var(--bad) 16%,transparent);border:1px solid color-mix(in srgb,var(--bad) 40%,transparent)}}
@keyframes in{{from{{opacity:0;transform:translateX(-12px)}}to{{opacity:1;transform:none}}}}</style>
<div>{rows}</div><p class="muted">Approximate rebuild time: <b>{'about ' + str(round(rebuild/60)) + ' min' if rebuild > 90 else str(rebuild) + ' s'}</b>.
Rule: every layer after the first changed one is rebuilt, which is why code is copied last.</p>""", height=420)

# ── 3. Manual steps ───────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Step by step, by hand")
    st.caption("Tick each step as you go — progress is kept for this browser session.")
    steps = [
        ("Start Docker Desktop and check the engine", "docker info", "Shows `Server Version: …`. If you see `dockerDesktopLinuxEngine: cannot find the file`, Docker Desktop isn't running yet."),
        ("Build the image from the project root", "docker build -t camel-sentinel:v2.1 .", "~10 min the first time (torch download), seconds for code-only changes afterwards."),
        ("Look at what you built", "docker images camel-sentinel\ndocker history camel-sentinel:v2.1", "One row per layer; the pip layer is ~2 GB."),
        ("Run it (with your secrets, at run time)", "docker run -d --name camel -p 8501:8501 -p 8000:8000 -e CAMEL_API_PUBLIC=1 --env-file .env camel-sentinel:v2.1", "CAMEL_API_PUBLIC=1 only for this class demo, so you can curl the API; by default the API is private to the container. Omit `--env-file .env` if you have no Azure keys; only chat/voice needs them."),
        ("Watch it become healthy", "docker ps\ndocker logs -f camel", "STATUS goes `(health: starting)` → `(healthy)` in ~1–2 min."),
        ("Test the API and open the UI", "curl http://localhost:8000/health\n# then open http://localhost:8501", '`{"status":"ok","model_loaded":true}`'),
        ("Prove the security basics", "docker exec camel whoami\ndocker exec camel ls -a /app", "`camel` (not root), and no `.env` in /app."),
        ("Log in and push to Docker Hub", f"docker login -u raushanranjan\ndocker tag camel-sentinel:v2.1 {DOCKERHUB_REPO}:v2.1\ndocker push {DOCKERHUB_REPO}:v2.0", "Type your password yourself, never into an AI chat."),
        ("Create ACR and push there too", f"az login\naz group create -n {RG} -l centralindia\naz acr create -g {RG} -n camelsentinelacr --sku Basic\naz acr login -n camelsentinelacr\ndocker tag camel-sentinel:v2.0 {ACR_REPO}:v2.0\ndocker push {ACR_REPO}:v2.0", "ACR Basic ≈ US$5/month while it exists."),
        ("Clean up locally and pull back from the registry", f"docker stop camel && docker rm camel\ndocker rmi camel-sentinel:v2.1 {DOCKERHUB_REPO}:v2.1 {ACR_REPO}:v2.0\ndocker run -d -p 8501:8501 {DOCKERHUB_REPO}:v2.0", "The last command downloads the image from Docker Hub, which proves the published copy works."),
    ]
    done = 0
    for i, (title, cmd, expect) in enumerate(steps, 1):
        with st.container(border=True):
            ck = st.checkbox(f"**Step {i}. {title}**", key=f"deploy_step_{i}")
            done += ck
            st.code(cmd, language="bash")
            st.caption(f"Expected: {expect}")
    st.progress(done / len(steps), text=f"{done}/{len(steps)} steps done")

# ── 4. With AI ────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("The same result with an AI coding assistant")
    st.markdown("Give the assistant **context**, one **goal**, and **constraints**, then *review* what it produces. These are the prompts used to build this very setup:")
    prompts = [
        ("Plan first", "Read backend/main.py, frontend/app.py, requirements.txt and docs/. Before writing any code, write docs/06_deployment_containerization.md describing how to package this FastAPI + Streamlit + ML-model project as ONE Docker image, publish it to Docker Hub and Azure Container Registry, and run it in the cloud.", "Does the plan keep secrets out of the image? Does it name the ports, the models and the data the app needs?"),
        ("Dockerfile", "Write a production Dockerfile: python:3.10-slim, CPU-only torch pinned as torch==2.13.0+cpu, requirements pinned to the versions the models were trained with (scikit-learn 1.7.2), dependencies copied before code for layer caching, a non-root user with COPY --chown, a HEALTHCHECK on /health, and no .env in the image.", "Is there a USER line? Is .env excluded? Is torch the +cpu build?"),
        (".dockerignore", "Write a .dockerignore that excludes .env and other secrets, virtualenvs, __pycache__, zip archives, Project-versions/, raw data, and notebooks' heavy outputs.", "Open it and look for .env on the first lines."),
        ("Entrypoint", "Write docker/docker-entrypoint.sh that starts uvicorn and streamlit together when MODE=all, or only one of them for MODE=api / MODE=ui. It must use LF line endings.", "Test all three modes."),
        ("Build + verify", "Build the image, run it with --env-file .env, wait until docker ps shows (healthy), then verify /health returns model_loaded=true, the UI returns 200, whoami is not root, and /app has no .env.", "Read the actual command output, not just the AI's summary."),
        ("Publish", "Tag and push the image to Docker Hub as raushanranjan/camel-sentinel:v2.0 and :latest, create resource group rg-camel-sentinel and ACR camelsentinelacr (Basic), push the same tags there, then remove the local images and pull from Docker Hub to prove it works.", "Check the tags in the registry web UI."),
        ("Scan", "Scan the Dockerfile and deploy/Dockerfile.naive with Trivy (docker run aquasec/trivy config) and explain each finding and its fix.", "Compare the before/after findings."),
    ]
    for title, prompt, review in prompts:
        with st.expander(f"🧩 {title}"):
            st.code(prompt, language="text")
            st.markdown(f"**Review checkpoint:** {review}")
    st.warning("Never paste passwords, access keys or `.env` contents into an AI assistant. Run `docker login` / `az login` yourself.")

# ── 5. Registries ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Where the image lives")
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown(f"### 🐳 Docker Hub\n`{DOCKERHUB_REPO}`\n\nPublic, free, the default registry for `docker pull`.\n\n[Open on Docker Hub ↗](https://hub.docker.com/r/{DOCKERHUB_REPO})")
        st.code(f"docker pull {DOCKERHUB_REPO}:latest\ndocker run -d -p 8501:8501 {DOCKERHUB_REPO}:latest", language="bash")
    with b, st.container(border=True):
        st.markdown(f"### 🔷 Azure Container Registry\n`{ACR_REPO}`\n\nPrivate, in resource group `{RG}` (Central India). Sits next to Azure compute; pulls use Azure identity, not passwords.")
        st.code(f"az acr login -n camelsentinelacr\ndocker pull {ACR_REPO}:latest\naz acr repository show-tags -n camelsentinelacr --repository camel-sentinel -o table", language="bash")
    st.subheader("Tags published")
    st.table([
        {"Tag": "v1.0", "Contents": "The app before the Deploy / Security Lab / Roadmap pages — the safe backup"},
        {"Tag": "v2.0", "Contents": "Adds Deploy / Security Lab / Roadmap pages, COPY --chown layout (smaller)"},
        {"Tag": "v2.1", "Contents": "Security-hardened (audit fixes, upgraded web stack, no curl), credits, grouped navigation, Security Posture page"},
        {"Tag": "v2.2", "Contents": "Switchable backend API (sidebar / ?backend= link, allow-listed + verified); one-go Container Apps guide"},
        {"Tag": "v2.3", "Contents": "Open-source release matching the GitHub repo; accurate SHAP base-rate display"},
        {"Tag": "latest", "Contents": "Always the newest pushed version (now v2.3)"},
    ])
    st.markdown("**Tag vs digest:** a tag (`latest`) can be moved to a new image; a digest (`@sha256:…`) never changes. Production deployments pin digests.")

# ── 6. Cloud ──────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("⭐ The whole app in one go — one Container App, image from ACR")
    st.markdown("One container runs **both** the API and the UI (`MODE=all`). Expose **port 8501** (the UI); the UI "
                "talks to the API inside the same container, so the API never needs to be public.")
    st.code(f"""# already created once: {RG} + ACR camelsentinelacr + a Container Apps environment
az containerapp create -g {RG} -n camel-sentinel \\
  --environment <your-containerapps-environment> \\
  --image {ACR_REPO}:latest \\
  --registry-server {ACR_SERVER} --registry-identity system \\
  --target-port 8501 --ingress external \\
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1

# an app that already exists (e.g. one exposing port 8000) → switch it to the full app:
az containerapp update  -g {RG} -n <app> --image {ACR_REPO}:latest --cpu 2 --memory 4Gi
az containerapp ingress update -g {RG} -n <app> --target-port 8501

az containerapp show -g {RG} -n <app> --query properties.configuration.ingress.fqdn -o tsv""", language="bash")
    st.markdown("""
- **Why 8501, not 8000:** 8000 is the API (you'd see Swagger at `/docs`); 8501 is the Streamlit UI. From v2.1 the API only
  listens inside the container in `MODE=all`, so exposing 8000 of a `MODE=all` app shows nothing.
- **Why 2 CPU / 4 GiB:** the image loads torch, two models, SHAP + LIME explainers and the 18 MB panel. 0.5 CPU / 1 GiB is too tight.
- **Chat / voice:** add your keys as secrets: `az containerapp secret set … --secrets aoai-key=<key>` then
  `az containerapp update … --set-env-vars AZURE_OPENAI_KEY=secretref:aoai-key AZURE_OPENAI_ENDPOINT=<endpoint>`.
- **Registry login:** `--registry-identity system` lets the app pull from ACR with its managed identity (no passwords).
""")

    st.subheader("🔌 Test a cloud API from your local UI")
    st.markdown("Keep an **API-only** Container App (`MODE=api`, target port 8000) and point any UI at it. The sidebar's "
                "**🔌 Backend API** panel switches every page at once. Or open one link that connects straight away:")
    st.code("""# API-only app in Azure (Swagger at https://<api-app>/docs)
az containerapp update -g rg-camel-sentinel -n <api-app> --image camelsentinelacr.azurecr.io/camel-sentinel:latest \\
  --set-env-vars MODE=api
az containerapp ingress update -g rg-camel-sentinel -n <api-app> --target-port 8000

# local UI (no local API needed), connected in one click:
docker run -d -p 8501:8501 -e MODE=ui raushanranjan/camel-sentinel:latest
# then open:  http://localhost:8501/?backend=https://<api-app>.azurecontainerapps.io""", language="bash")
    st.caption("Safety: only allow-listed hosts (localhost, compose `api`, *.azurecontainerapps.io, *.azurewebsites.net, "
               "*.azure-api.net, plus CAMEL_BACKEND_ALLOWLIST) that answer /health like a CAMEL API are accepted. "
               "CAMEL_LOCK_BACKEND=1 pins the address for production.")

    st.subheader("Option A — Azure Container Apps from Docker Hub (simplest, scales to zero)")
    st.code(f"""az extension add --name containerapp --upgrade
az containerapp env create -g {RG} -n camel-env -l centralindia

az containerapp create -g {RG} -n camel-sentinel --environment camel-env \\
  --image docker.io/{DOCKERHUB_REPO}:latest \\
  --target-port 8501 --ingress external \\
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1

az containerapp show -g {RG} -n camel-sentinel --query properties.configuration.ingress.fqdn -o tsv""", language="bash")
    st.markdown("""
- `--target-port 8501`: the public HTTPS URL forwards to Streamlit; the API stays internal on 8000.
- `--min-replicas 0`: scales to zero when idle, so you pay almost nothing between classes (first request then takes ~1–2 min to wake).
- Secrets: `--secrets aoai-key=<key> --env-vars AZURE_OPENAI_KEY=secretref:aoai-key`, never in the image.
- From ACR instead: `--image {acr}:latest --registry-server {srv} --registry-identity system`.
""".format(acr=ACR_REPO, srv=ACR_SERVER))
    st.subheader("Option B — docker compose (two services, one image)")
    st.code(read_file("docker-compose.yml"), language="yaml")
    st.subheader("Option C — Kubernetes (3 replicas, probes, service)")
    st.code(read_file("deploy/k8s/camel-sentinel.yaml"), language="yaml")
    st.caption("`readinessProbe`: don't send traffic until models are loaded. `livenessProbe`: restart a pod that stops answering. `runAsNonRoot`: refuse root images.")
    st.info(f"💸 When the course is finished: `az group delete -n {RG}` removes the registry and any Container Apps in one command.")

# ── 7. DevSecOps ──────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Trivy: scan before you ship")
    st.markdown("Trivy checks Dockerfiles for misconfigurations and images for known CVEs. Compare the deliberately naive file with ours:")
    a, b = st.columns(2)
    with a:
        st.markdown("**❌ deploy/Dockerfile.naive**")
        st.code(read_file("deploy/Dockerfile.naive"), language="docker")
    with b:
        st.markdown("**✅ Dockerfile (this project)**")
        st.code("\n".join(l for l in read_file("Dockerfile").splitlines() if l and not l.startswith("#")), language="docker")
    trivy_path = PROJECT_ROOT / "deploy" / "trivy_results.json"
    if trivy_path.exists():
        res = json.loads(trivy_path.read_text(encoding="utf-8"))
        st.markdown(f"**Real scan results** (`trivy config`, {res.get('scanned', '')}):")
        st.dataframe(res["findings"], width="stretch", hide_index=True)
        st.warning("Notice what Trivy did **not** flag: the naive file's `COPY . .` would copy `.env` (your API keys) "
                   "into the image. Scanners are a safety net, not a replacement for `.dockerignore` and code review.")
    st.code("docker run --rm -v \"%cd%:/scan\" aquasec/trivy config /scan          # Windows cmd\ndocker run --rm -v \"$PWD:/scan\" aquasec/trivy config /scan          # macOS / Linux\ndocker run --rm aquasec/trivy image raushanranjan/camel-sentinel:v2.0   # CVEs in packages", language="bash")

    st.subheader("Secrets, identity and transport")
    x, y, z = st.columns(3)
    with x, st.container(border=True):
        st.markdown("**🔑 Secrets**\n\n- `.env` excluded by `.dockerignore`\n- injected at run time (`--env-file`, Container Apps `secretref:`, K8s `Secret`)\n- production: Azure Key Vault / HashiCorp Vault, with rotation")
    with y, st.container(border=True):
        st.markdown("**👤 Least privilege**\n\n- `USER camel` (uid 10001)\n- K8s `runAsNonRoot`, `allowPrivilegeEscalation: false`\n- no compilers or shells beyond `sh` needed at run time")
    with z, st.container(border=True):
        st.markdown("**🔒 TLS / mTLS**\n\n- Container Apps ingress gives HTTPS automatically\n- mTLS = *both* sides show certificates (service mesh, API gateway)\n- keeps a stolen API key from being enough on its own")
    st.markdown("➡️ Attacks on this deployed system are demonstrated in the **🛡️ AI Security Lab** page (supply chain, model stealing, prompt injection…).")

# ── 8. Behind the scenes ──────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("What the image is really made of")
    st.markdown("Real layers of `camel-sentinel:v1.0` from `docker history`. Width ∝ size (log scale so small layers stay visible).")
    import math
    colors = {"base": "var(--muted)", "system": "var(--teal)", "deps": "var(--violet)", "code": "var(--accent)", "assets": "var(--ok)", "waste": "var(--bad)"}
    bars = "".join(
        f'<div class="row" style="animation-delay:{i*0.07}s"><span class="nm">{n}</span>'
        f'<span class="b" style="width:{max(4, math.log10(mb*100+1)/math.log10(199001)*100):.0f}%;background:{colors[k]}"></span>'
        f'<span class="mb">{mb:,.2f} MB</span></div>'
        for i, (n, mb, k) in enumerate(V1_LAYERS))
    html(f"""<style>.row{{display:grid;grid-template-columns:minmax(120px,38%) 1fr 80px;align-items:center;gap:8px;margin:4px 0;
opacity:0;animation:in .5s forwards;font-size:12.5px}}.b{{height:14px;border-radius:4px;display:block}}.mb{{text-align:right;color:var(--muted)}}
@keyframes in{{from{{opacity:0}}to{{opacity:1}}}}</style>{bars}
<p class="muted" style="margin-top:10px">Red = the <code>chown -R</code> layer: changing file ownership rewrites every file into a new layer.
v2.0 uses <code>COPY --chown</code> instead and drops it. Compressed push size ≈ 652 MB; unpacked ≈ 2.2 GB.</p>""", height=470)

    st.subheader("How isolation works")
    st.table([
        {"Mechanism": "Namespaces", "What it isolates": "Process IDs, network, mounts, hostname, users — the container sees only its own", "You notice it when": "`docker exec camel ps` shows just uvicorn + streamlit"},
        {"Mechanism": "cgroups", "What it isolates": "CPU and memory limits", "You notice it when": "`--memory 4g` or K8s `limits` stop one container starving others"},
        {"Mechanism": "Union filesystem", "What it isolates": "Read-only image layers + one thin writable layer per container", "You notice it when": "Files written inside vanish when the container is removed"},
        {"Mechanism": "Content addressing", "What it isolates": "Layers identified by SHA-256 of their content", "You notice it when": "`docker push` says `Layer already exists` for unchanged layers"},
    ])
    st.markdown("Containers share the host's kernel (unlike VMs), which is why they start in seconds. On Windows/macOS, Docker Desktop runs a small Linux VM for that kernel.")

# ── 9. Troubleshooting ────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Problems we actually hit — and the fixes")
    st.table([
        {"Symptom": "open //./pipe/dockerDesktopLinuxEngine: cannot find the file", "Cause": "Docker Desktop not running", "Fix": "Start Docker Desktop; wait for 'Engine running'"},
        {"Symptom": "No matching distribution found for flit_core (torch step)", "Cause": "--index-url pointed only at the PyTorch index", "Fix": "torch==2.13.0+cpu with --extra-index-url https://pypi.org/simple"},
        {"Symptom": "Image 23.7 MB bigger than needed", "Cause": "chown -R after COPY duplicates files", "Fix": "Create the user first, then COPY --chown"},
        {"Symptom": "exec docker-entrypoint.sh: no such file or directory", "Cause": "Windows CRLF line endings in the script", "Fix": "sed -i 's/\\r$//' in the Dockerfile + .gitattributes eol=lf"},
        {"Symptom": "denied: requested access to the resource is denied", "Cause": "Not logged in, or tag lacks your username", "Fix": "docker login; tag as raushanranjan/…"},
        {"Symptom": "Stuck at (health: starting)", "Cause": "Models + SHAP/LIME still loading", "Fix": "Wait ≤2 min; docker logs camel"},
        {"Symptom": "(healthy) but the API doesn't answer yet", "Cause": "Health check 'API || UI' passed as soon as Streamlit started", "Fix": "docker/healthcheck.sh checks only what the MODE owns: model_loaded=true"},
        {"Symptom": "UI: Can't reach the backend (compose)", "Cause": "CAMEL_BACKEND_URL=localhost inside the ui container", "Fix": "Use the service name: http://api:8000"},
        {"Symptom": "Chat says 'not configured'", "Cause": "No Azure keys passed at run time", "Fix": "--env-file .env"},
    ])
    st.caption("Full runbook: docs/06_deployment_containerization.md")
