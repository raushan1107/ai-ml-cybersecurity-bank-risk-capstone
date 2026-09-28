# Deployment & Containerization — Design and Runbook

> Learning path: **MLOps & On-Premises Deployment** (step 10 of the
> [AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)).
> This doc is the reference behind the **🚢 Deploy & Containerize** page in the app.

---

## 1. The goal

Take the whole CAMEL Sentinel project (FastAPI backend, Streamlit UI, the trained
models, the cached training panel and GNN artifacts), package it as **one Docker
image**, and publish that image to two registries:

| Registry | Image reference | Why both |
|---|---|---|
| Docker Hub | `raushanranjan/camel-sentinel:<tag>` | Public, free, the registry most tutorials assume |
| Azure Container Registry (ACR) | `camelsentinelacr.azurecr.io/camel-sentinel:<tag>` | Private, sits next to Azure compute (Container Apps, AKS, App Service) |

Anyone can then run the whole app with a single `docker run`, on a laptop or in the cloud,
without installing Python, torch, or any dependency.

Tags:

- `v1.0`: the app as it was before the Deploy / Security Lab / Roadmap pages were added (the safe backup).
- `v2.0`: adds the Deploy / Security Lab / Roadmap pages.
- `v2.1`: security-hardened release (see `docs/09_security_posture.md`), credits, grouped navigation, Security Posture page.
- `latest`: always points to the newest pushed version.

---

## 2. Concepts in one screen

| Term | Plain meaning | Analogy |
|---|---|---|
| **Dockerfile** | A text recipe: start from this base, copy these files, install these packages, run this command | A recipe card |
| **Image** | The read-only result of running the recipe: filesystem + metadata, built in **layers** | A frozen, vacuum-sealed meal |
| **Layer** | One cached filesystem diff per Dockerfile instruction; unchanged layers are reused | Each prep step saved separately, so editing the garnish doesn't re-cook the rice |
| **Container** | A running instance of an image, with its own isolated process, network and writable layer | The meal heated and served, where you can serve many from the same frozen batch |
| **Registry** | A server that stores and distributes images (Docker Hub, ACR, GHCR) | The supermarket freezer aisle |
| **Tag** | A human label on an image version (`v1.0`, `latest`) | The label on the box |
| **Digest** | The immutable SHA-256 content hash of an image | The barcode: it never lies |

What actually happens behind the scenes:

1. `docker build` sends the **build context** (the project folder minus `.dockerignore`) to the Docker engine.
2. The engine executes each instruction in a temporary container and snapshots the result as a layer.
3. `docker push` uploads only the layers the registry doesn't already have (content-addressed by digest).
4. `docker pull` / `docker run` on another machine downloads those layers, stacks them with a union filesystem, adds a thin writable layer, and starts the process in isolated Linux namespaces with cgroup resource limits.

---

## 3. Architecture of the image

```
┌──────────────────────── container: camel-sentinel ────────────────────────┐
│  docker-entrypoint.sh  (MODE = all | api | ui)                            │
│                                                                           │
│   uvicorn backend/main.py  :8000  ◄──── HTTP ────  streamlit frontend :8501│
│        │ loads                                        ▲                   │
│        ▼                                              │ browser           │
│   models/*.joblib, *.pt, *.pkl                        │                   │
│   data/processed/camel_panel.parquet, data/gnn/*.parquet                  │
│                                                                           │
│   runs as non-root user "camel" · HEALTHCHECK → GET :8000/health          │
└───────────────────────────────────────────────────────────────────────────┘
          secrets (Azure keys) come in at RUN time via --env-file, never baked in
```

**One image, three modes** (selected by the `MODE` environment variable):

| MODE | Starts | Use it when |
|---|---|---|
| `all` (default) | backend in background + UI in foreground | Single-container cloud targets (Azure Container Apps, App Service, ACI), classroom demo |
| `api` | backend only on `:8000` | Scaling the API separately, Kubernetes |
| `ui` | Streamlit only on `:8501`, calling `CAMEL_BACKEND_URL` | Two-container setup via `docker-compose.yml` |

The frontend reads `CAMEL_BACKEND_URL` (default `http://localhost:8000`), which is the
only code change containerization required.

---

## 4. Files added

| File | Purpose |
|---|---|
| `Dockerfile` | The recipe (python:3.10-slim, CPU-only torch, non-root user, healthcheck) |
| `.dockerignore` | Keeps `.env`, zips, `Project-versions/`, caches and notebooks' heavy outputs out of the build context |
| `requirements-docker.txt` | Runtime-only dependencies pinned to the versions the models were trained with (no Jupyter) |
| `docker/docker-entrypoint.sh` | Starts the right process(es) for `MODE` |
| `docker/healthcheck.sh` | Health = the service this `MODE` owns really works |
| `docker-compose.yml` | Two-service setup (api + ui) from the same image |
| `deploy/k8s/camel-sentinel.yaml` | Kubernetes Deployment (3 replicas, liveness/readiness probes) + Service |
| `deploy/Dockerfile.naive` | Deliberately insecure version used for the Trivy before/after demo |

### Key Dockerfile decisions

| Decision | Why |
|---|---|
| `python:3.10-slim` base | Matches the dev Python version; slim is about 120 MB vs about 1 GB for the full image |
| CPU-only torch wheel from `download.pytorch.org/whl/cpu` | The default torch wheel bundles CUDA (about 2.5 GB extra) we never use |
| Pin `scikit-learn==1.7.2` | The `.joblib` models were pickled with it; a different version can fail to unpickle or silently change behaviour |
| Copy `requirements-docker.txt` **before** the source code | Dependency layer is cached; editing a `.py` file rebuilds in seconds, not minutes |
| `USER camel` (non-root) | If the app is compromised, the attacker is not root inside the container (Trivy rule DS002) |
| `HEALTHCHECK` via `docker/healthcheck.sh` (API must report `model_loaded: true`) | Docker, and every orchestrator, can tell "process running" from "service working" (Trivy rule DS026) |
| No `.env` in image, secrets via `--env-file` | Anyone who pulls a public image can read every file in it, including keys |
| `start-period=120s` on healthcheck | Startup loads 2 models, builds SHAP + LIME explainers, reads an 18 MB panel and attempts a live FDIC pull |

Expected image size: about 1.8–2.2 GB, mostly torch CPU (about 700 MB) plus scientific Python.

---

## 5. Runbook — manual steps

### 5.0 Prerequisites

- Docker Desktop installed **and running** (the whale icon is steady). Check with `docker info`.
- A Docker Hub account (`docker login`, which you type yourself; never paste passwords into an AI chat).
- For ACR: Azure CLI + `az login`, an active subscription.

### 5.1 Build and test locally

```bash
# from the project root (the folder that contains the Dockerfile)
docker build -t camel-sentinel:v1.0 .

# run it — secrets come from your local .env at RUN time
docker run -d --name camel -p 8501:8501 --env-file .env camel-sentinel:v2.1
# the API is private to the container by default; to demo it from your machine add:
#   -p 8000:8000 -e CAMEL_API_PUBLIC=1

docker ps                                   # STATUS shows (health: starting) → (healthy)
docker exec camel /usr/local/bin/healthcheck.sh; echo $?   # 0 = healthy
curl http://localhost:8000/health           # only with CAMEL_API_PUBLIC=1
# open http://localhost:8501 in the browser
docker logs -f camel                        # watch startup
docker stop camel && docker rm camel
```

Without a `.env` file, omit `--env-file .env`: scoring, XAI, RAI, GNN, the Security Lab
and the Roadmap all work, and only the chat/voice feature reports "not configured".

### 5.2 Push to Docker Hub

```bash
docker login -u raushanranjan
docker tag camel-sentinel:v1.0 raushanranjan/camel-sentinel:v1.0
docker tag camel-sentinel:v1.0 raushanranjan/camel-sentinel:latest
docker push raushanranjan/camel-sentinel:v1.0
docker push raushanranjan/camel-sentinel:latest
```

### 5.3 Create ACR and push

```bash
az login
az group create -n rg-camel-sentinel -l centralindia
az acr create -g rg-camel-sentinel -n camelsentinelacr --sku Basic
az acr login -n camelsentinelacr

docker tag camel-sentinel:v1.0 camelsentinelacr.azurecr.io/camel-sentinel:v1.0
docker push camelsentinelacr.azurecr.io/camel-sentinel:v1.0
az acr repository show-tags -n camelsentinelacr --repository camel-sentinel -o table
```

Alternative without local Docker: `az acr build -r camelsentinelacr -t camel-sentinel:v1.0 .`
builds in Azure directly from your source folder.

Cost note: ACR Basic is about US$5/month while it exists. Delete it with
`az group delete -n rg-camel-sentinel` when the course is done.

### 5.4 Clean up locally (and prove the registry copy works)

```bash
docker rmi camel-sentinel:v1.0 raushanranjan/camel-sentinel:v1.0 \
           raushanranjan/camel-sentinel:latest camelsentinelacr.azurecr.io/camel-sentinel:v1.0
docker image prune -f
docker run -d -p 8501:8501 raushanranjan/camel-sentinel:v1.0   # pulls from Docker Hub
```

### 5.5 Deploy to the cloud (Azure Container Apps)

```bash
az extension add --name containerapp --upgrade
az containerapp env create -g rg-camel-sentinel -n camel-env -l centralindia

# from Docker Hub (public image)
az containerapp create -g rg-camel-sentinel -n camel-sentinel \
  --environment camel-env \
  --image docker.io/raushanranjan/camel-sentinel:latest \
  --target-port 8501 --ingress external \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
  --secrets aoai-key=<your-key> \
  --env-vars AZURE_OPENAI_KEY=secretref:aoai-key AZURE_OPENAI_ENDPOINT=<endpoint> AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

# or from ACR (private image)
az containerapp create ... --image camelsentinelacr.azurecr.io/camel-sentinel:latest \
  --registry-server camelsentinelacr.azurecr.io --registry-identity system

az containerapp show -g rg-camel-sentinel -n camel-sentinel --query properties.configuration.ingress.fqdn -o tsv
```

`--min-replicas 0` scales to zero when idle, so you pay almost nothing between classes.
Streamlit needs WebSockets, which Container Apps ingress supports out of the box.

### 5.5a The whole app in one Container App, image from ACR (recommended)

One container runs both processes (`MODE=all`); expose **8501** (the UI). The UI calls the API over
`localhost` inside the container, so the API is never public.

```bash
az containerapp create -g rg-camel-sentinel -n camel-sentinel \
  --environment <containerapps-environment> \
  --image camelsentinelacr.azurecr.io/camel-sentinel:latest \
  --registry-server camelsentinelacr.azurecr.io --registry-identity system \
  --target-port 8501 --ingress external --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1

# convert an existing app that exposes 8000 (API/Swagger only) into the full app:
az containerapp update -g rg-camel-sentinel -n <app> --image camelsentinelacr.azurecr.io/camel-sentinel:latest --cpu 2 --memory 4Gi
az containerapp ingress update -g rg-camel-sentinel -n <app> --target-port 8501
```

Common mistakes (seen for real on this project):

- **Target port 8000** shows only Swagger. From v2.1, a `MODE=all` app's API listens only inside the container, so port 8000 shows nothing at all. Use 8501, or run an API-only app with `MODE=api`.
- **0.5 CPU / 1 GiB** is too small for torch + two models + SHAP/LIME. Use 2 CPU / 4 GiB.

### 5.5b A cloud API with a local UI (for testing)

Run an API-only app (`--set-env-vars MODE=api`, target port 8000). Then point any UI at it:

```bash
docker run -d -p 8501:8501 -e MODE=ui raushanranjan/camel-sentinel:latest
# open http://localhost:8501/?backend=https://<api-app>.azurecontainerapps.io
```

Or paste the URL into the sidebar **🔌 Backend API** panel; the setting applies to every page. Only allow-listed hosts that
answer `/health` like a CAMEL API are accepted (see `docs/09`, F1). Chat and voice run *in the API*, so the
Azure keys must be set on the API app.

| Variable | Default | Meaning |
|---|---|---|
| `CAMEL_BACKEND_URL` | `http://localhost:8000` | Default API address for the UI |
| `CAMEL_BACKEND_ALLOWLIST` | empty | Extra allowed host patterns, comma-separated (e.g. `*.mycorp.net`) |
| `CAMEL_LOCK_BACKEND` | `0` | `1` hides the switch and pins the default address (production) |
| `CAMEL_API_PUBLIC` | `0` | `1` makes the API listen on all interfaces in `MODE=all` |

### 5.6 Kubernetes (concept + manifest)

`deploy/k8s/camel-sentinel.yaml` runs 3 replicas with liveness/readiness probes on
`/health` and a `LoadBalancer` Service. On any cluster (AKS, minikube, Docker Desktop's
built-in Kubernetes): `kubectl apply -f deploy/k8s/camel-sentinel.yaml`.

---

## 6. DevSecOps controls

| Control | Where in this project | Demo |
|---|---|---|
| **Image scanning (Trivy)** | `trivy config deploy/Dockerfile.naive` vs `trivy config Dockerfile` | Real scan (`deploy/trivy_results.json`): naive file fails DS-0002 root user (HIGH), DS-0001 `:latest` tag (MEDIUM), DS-0026 no HEALTHCHECK (LOW); ours passes 27/27. Trivy did *not* flag `COPY . .` copying `.env`, so scanners don't replace `.dockerignore` |
| **Secrets management** | `.env` excluded by `.dockerignore`; runtime `--env-file`; Container Apps `secretref:`; Key Vault / HashiCorp Vault for production | `docker history` / `docker run ... cat /app/.env` on the image shows no secrets |
| **Least privilege** | `USER camel`, no build tools in the final image | `docker exec camel whoami` → `camel` |
| **Transport security (TLS / mTLS)** | Container Apps ingress terminates HTTPS automatically; mTLS between services is a mesh / ingress setting | Explained on the page; not required to run the demo |

---

## 7. Prompts that produce the same result with an AI coding assistant

1. *"Read backend/main.py, frontend/app.py and requirements.txt. Write a production Dockerfile for this
   FastAPI + Streamlit project: python:3.10-slim, CPU-only torch, non-root user, HEALTHCHECK on /health,
   dependency layer cached before source copy. Do not copy .env into the image."*
2. *"Write a .dockerignore that excludes secrets, virtualenvs, caches, zip archives and version backups."*
3. *"Write an entrypoint script that starts uvicorn and streamlit together when MODE=all, or one of them
   for MODE=api / MODE=ui."*
4. *"Build the image, run it with --env-file .env, and verify /health returns 200 and the UI loads."*
5. *"Tag and push the image to Docker Hub as raushanranjan/camel-sentinel:v1.0, then create an Azure
   Container Registry in a new resource group and push the same image there."*
6. *"Scan both Dockerfiles with Trivy and explain every finding."*

Review checkpoints: after every step, check that the image contains no `.env`, runs as a non-root
user, becomes healthy, and that the pushed tag appears in the registry.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `open //./pipe/dockerDesktopLinuxEngine: cannot find the file` | Docker Desktop is not running | Start Docker Desktop, wait for "Engine running" |
| `denied: requested access to the resource is denied` on push | Not logged in, or tag doesn't start with your username | `docker login`; tag as `raushanranjan/...` |
| Container stuck at `(health: starting)` | Model/SHAP/LIME startup still running | Wait up to 2 min; `docker logs camel` |
| `(healthy)` but the API doesn't answer yet | The first health check was `curl API \|\| curl UI`, which passed as soon as Streamlit started (we hit this for real) | `docker/healthcheck.sh` checks only what the MODE owns: `/health` must report `model_loaded: true` |
| UI says "Can't reach the backend" in `ui` mode | `CAMEL_BACKEND_URL` wrong | In compose it's `http://api:8000` (the service name), not localhost |
| `InconsistentVersionWarning` / unpickle error | scikit-learn version differs from training | Keep `scikit-learn==1.7.2` pinned |
| Chat page "not configured" | No Azure keys passed at run time | Add `--env-file .env` |
| Image is huge (> 4 GB) | CUDA torch wheel installed | Use the CPU index URL line in the Dockerfile |
| `No matching distribution found for flit_core` during the torch step | `--index-url` pointed *only* at the PyTorch index, so pure-Python deps (typing-extensions) were looked up there | Pin `torch==X.Y.Z+cpu` and add `--extra-index-url https://pypi.org/simple` (we hit this for real while building this image) |
