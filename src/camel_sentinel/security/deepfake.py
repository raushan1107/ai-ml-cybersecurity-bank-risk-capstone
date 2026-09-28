"""Deepfake voice detection — a transparent, SYNTHETIC teaching simulation.

Real voice-clone detectors (and the ASVspoof research challenges) learn from
large labelled audio corpora. Here we generate 1-second voice-like signals so
learners can *see* the kinds of cues detectors use:

* **jitter**  — cycle-to-cycle wobble in pitch. Human vocal folds are never
  perfectly steady; older or cheaper voice clones are often too smooth.
* **shimmer** — cycle-to-cycle wobble in loudness, same idea.
* **high-band energy** — many vocoders band-limit output (e.g. 8 kHz audio
  upsampled from a 4–5 kHz synthesis band), leaving little energy up high.
* **spectral flatness** — breath and turbulence noise make human speech
  slightly "noisier" between harmonics.

A logistic-regression detector is trained on human voices vs *basic* clones,
the fakes that existed when it was built. The "advanced clone" preset mimics
human statistics and was never in training, to make the key lesson visible:
detectors don't generalise to newer generators, so banks
must pair them with *process* controls (call-back verification, dual
approval, provenance such as C2PA Content Credentials / SynthID watermarks).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SR = 16000
DURATION = 1.0

PRESETS: dict[str, dict[str, float]] = {
    #                 jitter  shimmer  band cutoff (Hz)  breath noise
    "human":          {"jitter": 0.012, "shimmer": 0.10, "cutoff": 7800, "noise": 0.030},
    "clone_basic":    {"jitter": 0.001, "shimmer": 0.01, "cutoff": 4000, "noise": 0.004},
    "clone_advanced": {"jitter": 0.010, "shimmer": 0.09, "cutoff": 7400, "noise": 0.026},
}
FEATURES = ["jitter_pct", "shimmer_pct", "high_band_ratio", "spectral_flatness"]


def synthesize(kind: str, seed: int = 0, jitter_scale: float = 1.0) -> np.ndarray:
    """A vowel-like source: harmonics of a wandering f0, amplitude-modulated, band-limited."""
    p = PRESETS[kind]
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    f0_base = rng.uniform(105, 210)
    n_cycles = int(f0_base * DURATION * 1.1) + 2
    # One period and one loudness per glottal cycle: that is what jitter/shimmer describe.
    periods = (1.0 / f0_base) * (1 + rng.normal(0, p["jitter"] * jitter_scale, n_cycles))
    amps = 1 + rng.normal(0, p["shimmer"], n_cycles)
    starts = np.concatenate([[0.0], np.cumsum(periods)])
    t = np.arange(n) / SR
    k = np.minimum(np.searchsorted(starts, t, side="right") - 1, n_cycles - 1)
    phase = 2 * np.pi * (t - starts[k]) / periods[k]  # 0 → 2π inside every cycle
    harmonics = np.arange(1, int((SR / 2) / f0_base))
    rolloff = 1.0 / harmonics ** 1.2
    # cos-based pulse: phase-aligned harmonics make each cycle a sharp glottal-like pulse
    signal = (rolloff[None, :] * np.cos(harmonics[None, :] * phase[:, None])).sum(axis=1)
    signal *= amps[k]
    signal += rng.normal(0, p["noise"] * 4, n)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / SR)
    spec[freqs > p["cutoff"]] *= 0.02
    out = np.fft.irfft(spec, n=len(signal))
    return out / (np.max(np.abs(out)) + 1e-9)


def _fundamental(x: np.ndarray) -> np.ndarray:
    """Keep only 60–320 Hz so zero crossings track the pitch, not the harmonics."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(freqs < 60) | (freqs > 320)] = 0
    return np.fft.irfft(spec, n=len(x))


def extract_features(x: np.ndarray) -> dict[str, float]:
    """Cycle statistics on the fundamental + band energy + flatness on the full signal."""
    f = _fundamental(x)
    rising = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    # sub-sample crossing times by linear interpolation
    frac = -f[rising] / (f[rising + 1] - f[rising] + 1e-12)
    crossings = (rising + frac) / SR
    periods = np.diff(crossings)
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100) if len(periods) > 3 else 0.0
    peaks = np.array([np.max(np.abs(f[a:b])) for a, b in zip(rising[:-1], rising[1:]) if b > a])
    shimmer = float(np.mean(np.abs(np.diff(peaks))) / np.mean(peaks) * 100) if len(peaks) > 3 else 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    high = float(power[freqs > 4500].sum() / (power.sum() + 1e-12))
    flat = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    return {"jitter_pct": round(jitter, 4), "shimmer_pct": round(shimmer, 4),
            "high_band_ratio": round(high, 6), "spectral_flatness": round(flat, 6)}


@lru_cache(maxsize=1)
def detector() -> Any:
    """Train once on human vs *basic* clones only — the fakes known when it was built."""
    X, y = [], []
    for i in range(150):
        X.append(list(extract_features(synthesize("human", seed=1000 + i)).values())); y.append(0)
        X.append(list(extract_features(synthesize("clone_basic", seed=5000 + i)).values())); y.append(1)
    X_arr = np.log(np.array(X) + 1e-6)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X_arr, y)


def run_deepfake_lab(kind: str = "human", seed: int = 42) -> dict[str, Any]:
    if kind not in PRESETS:
        raise ValueError(f"kind must be one of {sorted(PRESETS)}")
    x = synthesize(kind, seed=seed)
    feats = extract_features(x)
    prob_fake = float(detector().predict_proba(np.log(np.array([list(feats.values())]) + 1e-6))[0][1])
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    bins = np.linspace(0, SR / 2, 81)
    band_db = [float(10 * np.log10(power[(freqs >= a) & (freqs < b)].mean() + 1e-12)) for a, b in zip(bins[:-1], bins[1:])]
    human_ref = extract_features(synthesize("human", seed=7))
    verdict = "likely synthetic" if prob_fake >= 0.7 else "uncertain — verify out of band" if prob_fake >= 0.3 else "likely human"
    return {
        "kind": kind, "synthetic_demo": True,
        "waveform": [round(float(v), 4) for v in x[:800]],  # first 50 ms
        "spectrum": {"freq_hz": [round(float(b), 1) for b in bins[:-1]], "db": [round(v, 2) for v in band_db]},
        "features": feats, "human_reference": human_ref,
        "detector": {"prob_fake": round(prob_fake, 4), "verdict": verdict,
                     "trained_on": "human vs basic clones only"},
        "ground_truth": "human" if kind == "human" else "synthetic",
        "fooled": (kind != "human") and prob_fake < 0.7,
    }
