"""Floor policy: fuse per-module probabilities into one decision per tick.

Decisions: respond, wait, ignore, yield. Every decision carries a reason code.
Audio is the only required input; every other signal is optional (None = absent).
Nothing about the person is kept past the session: patience lives on the instance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RESPOND, WAIT, IGNORE, YIELD = "respond", "wait", "ignore", "yield"

# still-talking kinds from the audio model / annotation -> reason code
_HOLD_REASONS = {
    "block": "mid_turn_block",
    "prolongation": "mid_turn_prolongation",
    "repetition": "mid_turn_repetition",
    "word_search": "word_search",
}

# expected-answer context -> patience multiplier (plan: digits/spelling/open get more, yes/no less)
CONTEXT_PATIENCE = {"digits": 1.6, "spelling": 1.6, "open": 1.3, "yesno": 0.7}


@dataclass
class Signals:
    t: float                              # seconds, monotonic
    user_speaking: bool                   # VAD
    agent_speaking: bool = False
    p_finished: float | None = None       # audio/meaning turn detector (Smart Turn, LiveKit)
    p_still_talking: float | None = None  # "still talking" audio model
    still_talking_kind: str | None = None # block | prolongation | repetition | word_search
    p_barge_in: float | None = None       # while agent speaks: real attempt vs backchannel
    p_addressed: float | None = None      # addressee model (gaze, wording, loudness)
    face_speaking: bool | None = None     # any visible face speaking; None = no camera
    mouth_moving: float | None = None     # visual: mouth still working during silence
    gaze_to_agent: float | None = None    # visual: gaze returned to agent
    expected: str | None = None           # digits | spelling | open | yesno
    hold_requested: bool = False          # user asked for a moment ("hold on")
    asked: bool = False                   # words: the latest transcript ends in a question mark


@dataclass
class Decision:
    decision: str
    reason: str
    p_finished: float | None = None
    p_still_talking: float | None = None
    p_addressed: float | None = None
    hold_until: float | None = None


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _not(p: float | None) -> float | None:
    return None if p is None else 1 - p


def fuse(probs: list[tuple[float | None, float]]) -> float | None:
    """Weighted log-odds sum over present (p, weight) pairs. Missing sources drop out."""
    present = [(p, w) for p, w in probs if p is not None]
    if not present:
        return None
    z = sum(w * _logit(p) for p, w in present) / sum(w for _, w in present)
    return 1 / (1 + math.exp(-z))


@dataclass
class FloorPolicy:
    # thresholds; per-deployment calibration = construct with different values
    silence_timeout: float = 0.5   # s of silence to respond when the signals are neutral (or absent)
    tau: float = 0.6               # s of extra wait per unit log-odds of "still talking"
    min_silence: float = 0.2       # never respond before this much silence
    hold_thr: float = 0.5          # reason code only: wait reads "mid_turn_block" etc. above this
    addressed_thr: float = 0.5
    barge_thr: float = 0.5
    max_hold: float = 3.0          # ceiling; after it, respond with a neutral "still listening" cue
    recovery_window: float = 1.5   # user resumes this soon after we responded = we cut them off
    max_patience: float = 2.0
    wait_model: object = None      # learned wait (turnheed.wait.LearnedWait); used when both audio signals are present
    fast_wait: float | None = None  # hand the turn to a words check (e.g. an LLM completion filter) this early...
    fast_p_still: float = 0.9      # ...unless the audio is this sure they are mid-turn...
    fast_off_after: int = 2        # ...and only until this caller has had this many long mid-turn pauses
    fast_long: float = 1.0         # (s) what counts as a long pause
    fast_off_resumes: int = 2      # ...or this many times they carried on right after a hand-over (cut-off recoveries)
    patience_decay: float = 0.5    # after a reply that was not a cut-off, keep this share of the extra patience

    # per-session state (discarded with the instance)
    patience: float = 1.0
    _speech_end: float | None = None
    _last_respond: float | None = None
    _responded_this_gap: bool = False
    _turn_addressed: float | None = None
    events: dict = field(default_factory=lambda: {"cutoff_recovery": 0, "hold_ceiling": 0})
    pauses: list = field(default_factory=list)  # this caller's resumed mid-turn pauses (s); session only

    def step(self, s: Signals) -> Decision:
        p_addr = fuse([(s.p_addressed, 1.0)])
        if s.face_speaking is False and s.user_speaking:
            p_addr = min(p_addr if p_addr is not None else 1.0, 0.1)
            off_camera = True
        else:
            off_camera = False

        # agent is talking: yield to real attempts, ignore backchannels
        if s.agent_speaking:
            if not s.user_speaking:
                return Decision(WAIT, "agent_speaking")
            if p_addr is not None and p_addr < self.addressed_thr:
                return Decision(IGNORE, "off_camera_voice" if off_camera else "side_talk", p_addressed=p_addr)
            if s.p_barge_in is None or s.p_barge_in >= self.barge_thr:
                return Decision(YIELD, "barge_in", p_addressed=p_addr)
            return Decision(IGNORE, "backchannel", p_addressed=p_addr)

        if self._last_respond is not None and s.t - self._last_respond > self.recovery_window:
            # the reply stood: that ending was real, so let session patience relax toward 1
            self._last_respond = None
            self.patience = 1 + (self.patience - 1) * self.patience_decay

        if s.user_speaking:
            if self._last_respond is not None and s.t - self._last_respond <= self.recovery_window:
                # we cut in and they kept going: stop, merge both parts into one turn, be more patient
                if self._speech_end is not None:
                    self.pauses.append(s.t - self._speech_end)  # a real mid-turn pause at least this long
                self._last_respond = None
                self.events["cutoff_recovery"] += 1
                self.patience = min(self.patience * 1.25, self.max_patience)
                self._speech_end, self._responded_this_gap = None, False
                return Decision(YIELD, "cutoff_recovery", p_addressed=p_addr)
            if self._speech_end is not None and not self._responded_this_gap and s.t - self._speech_end >= self.min_silence:
                self.pauses.append(s.t - self._speech_end)  # they paused and carried on: learn their pause length
            self._speech_end, self._responded_this_gap = None, False
            self._turn_addressed = p_addr
            if p_addr is not None and p_addr < self.addressed_thr:
                return Decision(IGNORE, "off_camera_voice" if off_camera else "side_talk", p_addressed=p_addr)
            return Decision(WAIT, "speaking", p_addressed=p_addr)

        # silence
        if self._speech_end is None:
            self._speech_end = s.t
        if self._responded_this_gap:
            return Decision(WAIT, "responded")
        if self._turn_addressed is not None and self._turn_addressed < self.addressed_thr:
            return Decision(IGNORE, "side_talk", p_addressed=self._turn_addressed)
        silence = s.t - self._speech_end
        patience = self.patience * CONTEXT_PATIENCE.get(s.expected or "", 1.0)
        hold_until = self._speech_end + self.max_hold * patience

        # every signal moves one wait time: log-odds of "still talking" from all present sources.
        # In-turn pauses end at a roughly constant rate (FluencyBank: ~0.6 s mean remaining past 0.5 s),
        # so the evidence for "finished" grows linearly with silence: wait = timeout + tau * log-odds.
        p_still = fuse([(s.p_still_talking, 1.0), (s.mouth_moving, 1.0)])
        p_cont = fuse([(p_still, 1.0), (_not(s.p_finished), 1.0), (_not(s.gaze_to_agent), 0.5)])
        z = _logit(p_cont) if p_cont is not None else 0.0
        if self.wait_model is not None and s.p_finished is not None and s.p_still_talking is not None:
            base = self.wait_model(s.p_finished, s.p_still_talking, tuple(self.pauses))
        else:
            base = self.silence_timeout + self.tau * z
        fast = (s.p_still_talking is not None and s.p_still_talking < self.fast_p_still
                and sum(x > self.fast_long for x in self.pauses) < self.fast_off_after
                and self.events["cutoff_recovery"] < self.fast_off_resumes)
        # a question ("are you there?") hands the turn over even from a caller who needs patience: a live call
        # held 7 s after one because the sound said "still talking" and the caller had asked for time
        if (self.fast_wait is not None and (fast or s.asked)
                and s.expected not in ("digits", "spelling")):  # numbers and spelling come in pieces: never rush
            # targeted fast path: only safe with a words veto downstream, which keeps a mid-sentence turn silent;
            # a caller with long mid-turn pauses (e.g. blocks) has it switched off for the rest of the call
            base = min(base, self.fast_wait / patience)
        wait = min(max(base, self.min_silence), self.max_hold) * patience
        d = dict(p_finished=s.p_finished, p_still_talking=p_still, p_addressed=p_addr,
                 hold_until=self._speech_end + wait)

        if s.t >= hold_until:
            self.events["hold_ceiling"] += 1
            return self._respond(s.t, "hold_ceiling", d)
        if s.hold_requested:
            return Decision(WAIT, "user_hold", **d)
        if silence >= wait:
            gaze = s.gaze_to_agent is not None and s.gaze_to_agent >= 0.5
            return self._respond(s.t, "gaze_returned" if gaze else "finished", d)
        if p_still is not None and p_still >= self.hold_thr:
            return Decision(WAIT, _HOLD_REASONS.get(s.still_talking_kind or "", "still_talking"), **d)
        return Decision(WAIT, "pause", **d)

    def _respond(self, t: float, reason: str, d: dict) -> Decision:
        self._last_respond, self._responded_this_gap = t, True
        return Decision(RESPOND, reason, **d)
