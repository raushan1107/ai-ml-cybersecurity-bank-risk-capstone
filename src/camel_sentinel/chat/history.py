"""
chat/history.py — rolling-window conversation history.

The backend is stateless: the Streamlit page (Slice B) holds the full
history in st.session_state and sends it with every request. This class
is used in two places:
  - backend /chat: validates and sanitizes incoming history before it
    reaches the OpenAI API (strips unknown keys, enforces allowed roles).
  - frontend chat page: manages the local session list (Slice B).

The rolling-window cap prevents unbounded context growth. MAX_TURNS=20
keeps the payload under ~4 k tokens for typical financial questions, well
within gpt-5-mini's context budget, while still allowing multi-step
follow-ups like "what if the NPL ratio were lower?".
"""

from __future__ import annotations

_ALLOWED_ROLES = frozenset({"user", "assistant"})


class ConversationHistory:
    MAX_TURNS = 20  # one turn = one user message + one assistant reply

    def __init__(self, messages: list[dict] | None = None) -> None:
        self._messages: list[dict] = []
        for m in messages or []:
            self.add(m["role"], m["content"])

    def add(self, role: str, content: str) -> None:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Role must be 'user' or 'assistant', got {role!r}")
        self._messages.append({"role": role, "content": str(content)})
        cap = self.MAX_TURNS * 2
        if len(self._messages) > cap:
            self._messages = self._messages[-cap:]

    def to_list(self) -> list[dict]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
