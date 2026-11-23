"""LiveKit Agents turn detector: wraps any LiveKit turn detector and holds the floor.

    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    from turnheed.plugins.livekit import FloorTurnDetector
    AgentSession(turn_detection=FloorTurnDetector(MultilingualModel(), still_talking=my_hold_signal))

LiveKit asks the detector for p(end of turn) from the chat context and waits its max endpointing
delay when p is below the unlikely threshold. Holding = pushing p down while the person is
mid-turn, so LiveKit's own max delay acts as the hold ceiling. The detector never sees audio,
so the hold signal comes from outside (the "still talking" model or the eyes module).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..policy import CONTEXT_PATIENCE


class FloorTurnDetector:
    def __init__(self, inner: Any, still_talking: Optional[Callable[[], Optional[float]]] = None):
        self._inner = inner
        self._still_talking = still_talking
        self.expected: str | None = None  # dialogue state: digits | spelling | open | yesno

    @property
    def model(self) -> str:
        return f"turnheed+{getattr(self._inner, 'model', 'unknown')}"

    @property
    def provider(self) -> str:
        return "turnheed"

    async def unlikely_threshold(self, language):
        return await self._inner.unlikely_threshold(language)

    async def supports_language(self, language) -> bool:
        return await self._inner.supports_language(language)

    async def predict_end_of_turn(self, chat_ctx, *, timeout: float | None = None) -> float:
        p = await self._inner.predict_end_of_turn(chat_ctx, timeout=timeout)
        p_still = self._still_talking() if self._still_talking else None
        if p_still is not None:
            p *= 1 - p_still
        # more patience while reading digits / spelling: same exponent trick keeps p in [0, 1]
        return p ** CONTEXT_PATIENCE.get(self.expected or "", 1.0)
