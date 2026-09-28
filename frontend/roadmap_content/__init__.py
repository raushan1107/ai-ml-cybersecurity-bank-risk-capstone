"""
Roadmap content — the whole learning path as data (docs/08_ai_ml_roadmap.md).

Each stage:  n, name, track, icon, tagline, handoff, topics[]
Each topic:  id, name, oneliner, real_world [[who, what], …], analogy, math,
             code, in_project, security

Stages are referred to by name with a plain sequence number — never "Module N".
Edit the stages_*.py files to change content; load_curriculum() validates it.
"""
from __future__ import annotations

from .stages_01_03 import STAGES as _A
from .stages_04_06 import STAGES as _B
from .stages_07_09 import STAGES as _C
from .stages_10_12 import STAGES as _D

STAGE_FIELDS = ("n", "name", "track", "icon", "tagline", "handoff", "topics")
TOPIC_FIELDS = ("id", "name", "oneliner", "real_world", "analogy", "math", "code", "in_project", "security")
TRACKS = ["Data & Models", "Language & GenAI", "Trust & Privacy", "Production & Security"]


def load_curriculum() -> list[dict]:
    stages = [*_A, *_B, *_C, *_D]
    seen: set[str] = set()
    for s in stages:
        missing = [f for f in STAGE_FIELDS if f not in s]
        assert not missing, f"stage {s.get('n')} missing {missing}"
        assert s["track"] in TRACKS, f"stage {s['n']} has unknown track {s['track']}"
        for t in s["topics"]:
            missing = [f for f in TOPIC_FIELDS if not t.get(f)]
            assert not missing, f"topic {t.get('id')} missing {missing}"
            assert t["id"] not in seen, f"duplicate topic id {t['id']}"
            seen.add(t["id"])
    assert [s["n"] for s in stages] == [str(i) for i in range(1, 13)], "stages must run 1..12 in order"
    return stages
