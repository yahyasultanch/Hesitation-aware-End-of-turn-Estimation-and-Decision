"""Pipecat turn analyzer: wraps Smart Turn (or any BaseTurnAnalyzer) with the floor policy.

    from turnheed.plugins.pipecat import FloorTurnAnalyzer
    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=FloorTurnAnalyzer())

One analyzer per call: session patience lives on it and is gone when it is.
Pipecat calls analyze_end_of_turn once when VAD stops; after that, a late end of turn
(hold ceiling, timeout) has to come out of append_audio, which is where the policy steps.
"""
from __future__ import annotations

import re

from typing import Optional, Protocol, Tuple

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

from ..policy import RESPOND, Decision, FloorPolicy, Signals

class StillTalking(Protocol):  # the "still talking" model plugs in here (turnheed.still_talking)
    def push(self, pcm16: bytes, sample_rate: int) -> None: ...    # every chunk: buffer only
    def read(self) -> Tuple[Optional[float], Optional[str]]: ...   # once per pause: (p_still_talking, kind)


class FloorTurnAnalyzer(BaseTurnAnalyzer):
    STT_LAG = 1.5  # s
    def __init__(self, *, inner: BaseTurnAnalyzer | None = None, policy: FloorPolicy | None = None,
                 still_talking: StillTalking | None = None, sample_rate: int | None = None):
        super().__init__(sample_rate=sample_rate)
        if inner is None:
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
            # the policy owns the timeout; keep Smart Turn's own stop_secs out of the way
            inner = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=600))
        self._inner = inner
        self.policy = policy or FloorPolicy()
        self._still_talking = still_talking
        self.expected: str | None = None       # dialogue state: digits | spelling | open | yesno
        self.last_decision: Decision | None = None
        self._t = 0.0                          # stream clock from sample count, not wall time
        self._heard = False
        self._p_finished: float | None = None
        self._complete = False
        self._last_still: tuple = (None, None)
        self.asked = False                     # latest transcript ends in "?" (set by FloorTurnStopStrategy)
        self._words_only = False               # turn known only from a transcript, not from the VAD
        self._last_speech = 0.0                # stream time of the last chunk the VAD called speech

    def set_sample_rate(self, sample_rate: int):
        super().set_sample_rate(sample_rate)
        self._inner.set_sample_rate(sample_rate)

    @property
    def speech_triggered(self) -> bool:
        return self._inner.speech_triggered

    @property
    def params(self):
        return self._inner.params

    def update_vad_start_secs(self, vad_start_secs: float):
        self._inner.update_vad_start_secs(vad_start_secs)

    def _step(self, is_speech: bool, buffer: bytes | None = None) -> bool:
        if self._still_talking and buffer is not None:
            self._still_talking.push(buffer, self.sample_rate or 16000)
            if is_speech:
                self._last_still = (None, None)
            elif self._last_still == (None, None):  # first silent chunk: read once, hold it for the pause
                self._last_still = self._still_talking.read()
        p_still, kind = self._last_still  # no new chunk on the VAD-stop call: reuse the pause's reading
        if self._words_only:  # speech the VAD missed: the sound reading is from an older pause, so drop it
            p_still, kind = None, None
        d = self.policy.step(Signals(t=self._t, user_speaking=is_speech, p_finished=self._p_finished,
                                     p_still_talking=p_still, still_talking_kind=kind, expected=self.expected,
                                     asked=self.asked))
        self.last_decision = d
        if d.decision == RESPOND:
            self._complete = True
        return self._complete

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        self._inner.append_audio(buffer, is_speech)
        self._t += len(buffer) / 2 / (self.sample_rate or 16000)
        if is_speech:
            self._heard, self._p_finished, self._complete, self.asked = True, None, False, False
            self._words_only, self._last_speech = False, self._t
        if not self._heard:
            return EndOfTurnState.INCOMPLETE
        return EndOfTurnState.COMPLETE if self._step(is_speech, buffer) else EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        if self._complete:  # append_audio already ended the turn; Pipecat calls us again for metrics
            self._reset_turn()
            return EndOfTurnState.COMPLETE, None
        state, metrics = await self._inner.analyze_end_of_turn()
        prob = getattr(metrics, "probability", None)
        self._p_finished = prob if prob is not None else (1.0 if state == EndOfTurnState.COMPLETE else 0.0)
        # ponytail: if the policy holds after Smart Turn said COMPLETE, Smart Turn has cleared its
        # buffer, so it judges the rest of the turn without the start. Fine for one re-check per turn.
        if self._step(False):
            self._reset_turn()
            return EndOfTurnState.COMPLETE, metrics
        return EndOfTurnState.INCOMPLETE, metrics

    def heard_speech(self):
        """Speech the VAD missed (a transcript arrived anyway): restart the silence clock from now."""
        # STT runs up to ~1.5 s behind the audio: a transcript that late is the pause the sound already read.
        # Later than that, the VAD missed the speech and the sound reading belongs to an older pause.
        stale = self._t - self._last_speech > self.STT_LAG
        self._heard, self._complete = True, False
        if stale:
            self._p_finished, self._words_only = None, True
        self.policy.step(Signals(t=self._t, user_speaking=True, expected=self.expected))

    def _reset_turn(self):
        self._heard, self._p_finished, self._complete = False, None, False
        self._last_still = (None, None)

    def clear(self):
        self._inner.clear()
        self._reset_turn()

    async def cleanup(self):
        await self._inner.cleanup()


BACKCHANNEL = {"okay", "ok", "yeah", "yep", "yes", "right", "sure", "alright", "huh"}
# hums and fillers however the STT spells them: mm, mhm, mhmm, mmhmm, mm-hmm, hmm, uh-huh, uh, um, ah, oh
HUM = re.compile(r"^(m+h*m*|h+m+|m+-?h+m+|u+h+(-?h+u+h+)?|u+m+|a+h+|o+h+)$")


def backchannel(word: str) -> bool:
    return not word or word in BACKCHANNEL or bool(HUM.match(word))


class FloorTurnStartStrategy(MinWordsUserTurnStartStrategy):
    """The floor policy's yield / ignore at the start of a turn: noise never takes the turn, words do.

    Stock Pipecat starts a user turn (and stops the agent mid-sentence) on any sound the VAD calls speech, or on any
    transcript. Callers almost always have background noise, breath, a cough; on a live call a half-second sound
    stopped the agent in the middle of reading out free slots and left 11 s of dead air. Here:
    - the VAD alone never starts a turn (sound with no words is ignored);
    - while the agent is speaking, a transcript of only backchannels ("mm", "okay", "yeah") is ignored; any other
      word yields the turn at once, so a caller who really wants to speak still interrupts.

        UserTurnStrategies(start=[FloorTurnStartStrategy()], stop=[FloorTurnStopStrategy(...)])
    """

    def __init__(self, **kwargs):
        super().__init__(min_words=1, **kwargs)

    async def _handle_transcription(self, frame):
        words = [w.strip(".,!?;:-").lower() for w in frame.text.split()]
        if self._bot_speaking and all(backchannel(w) for w in words):
            return ProcessFrameResult.CONTINUE
        return await super()._handle_transcription(frame)


class FloorTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Pipecat's turn-analyzer stop strategy, except when a transcript arrives that the VAD never heard.

    Stock Pipecat then "assumes the turn is complete" and ends it ~0.15 s later, bypassing the analyzer. Quiet,
    stuttered syllables are exactly what a VAD misses, so on a live test call that cut the caller off
    again and again. Here the transcript counts as speech and the floor policy decides when the turn ends.

        TurnAnalyzerUserTurnStopStrategy -> FloorTurnStopStrategy(turn_analyzer=FloorTurnAnalyzer(...))
    """

    async def _handle_transcription(self, frame):
        if isinstance(self._turn_analyzer, FloorTurnAnalyzer):
            self._turn_analyzer.asked = frame.text.rstrip().endswith("?")
        # Any transcript that lands while the VAD says "silent" is speech the VAD missed, or arrived late (STT runs
        # behind the audio). Either way the caller just spoke: restart the floor policy's silence clock and let it
        # end the turn. Stock Pipecat instead either ends the turn ~0.15 s later, or (after the analyzer already
        # said complete) hangs until its fallback timeout: 10 s of dead air on a live call after spoken digits.
        if not self._vad_user_speaking and isinstance(self._turn_analyzer, FloorTurnAnalyzer):
            self._text = frame.text
            if frame.finalized:
                self._transcript_finalized = True
            self._turn_complete = False
            self._turn_analyzer.heard_speech()
            return
        await super()._handle_transcription(frame)
