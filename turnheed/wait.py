"""Learned wait: how long to stay quiet after speech stops, from the evidence (bench/recipes/fit_wait.py).

    pol = FloorPolicy(wait_model=LearnedWait.load("data/still_talking/wait_model.json"))

pi = P(turn ended | p_finished, p_still_talking) from 3 logistic weights; S(s | evidence bin) = share of real
mid-turn pauses longer than s. Answer at the first s where pi / (pi + (1 - pi) S(s)) >= theta: quickly when
the evidence says finished, longer only where mid-turn pauses of that kind really run long. Pure Python.
"""
from __future__ import annotations

import bisect
import json
import math
from functools import lru_cache


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class LearnedWait:
    def __init__(self, beta, edges, grid, S, theta):
        self.beta, self.edges, self.grid, self.S, self.theta = beta, edges, grid, S, theta
        self._cached = lru_cache(maxsize=4096)(self._wait)

    @classmethod
    def load(cls, path: str, theta: float | None = None) -> "LearnedWait":
        d = json.load(open(path, encoding="utf8"))
        return cls(d["beta"], d["edges"], d["grid"], d["S"], d["theta"] if theta is None else theta)

    def _wait(self, p_finished: float, p_still: float) -> float:
        z = self.beta[0] * _logit(p_finished) + self.beta[1] * _logit(p_still) + self.beta[2]
        pi = 1 / (1 + math.exp(-z))
        surv = self.S[min(bisect.bisect_left(self.edges, z), len(self.S) - 1)]
        for s, S in zip(self.grid, surv):
            if pi / (pi + (1 - pi) * max(S, 1e-6)) >= self.theta:
                return s
        return self.grid[-1]

    def __call__(self, p_finished: float, p_still: float, session: tuple = ()) -> float:
        """session: this caller's own resumed mid-turn pause lengths (s), this call only."""
        if len(session) < 3:
            return self._cached(round(p_finished, 3), round(p_still, 3))  # signals are constant within a pause
        return self._session_wait(round(p_finished, 3), round(p_still, 3), tuple(round(x, 1) for x in session[-30:]))

    @lru_cache(maxsize=256)
    def _session_wait(self, p_finished, p_still, session):
        # blend the population "mid-turn pause lasts longer than s" curve with this caller's own pauses;
        # the population counts as PRIOR pauses, so a few long blocks move it and a fluent caller stays fast
        n = len(session)
        emp = [sum(x > g for x in session) / n for g in self.grid]
        # upward only: a caller's own pauses can earn longer waits, never shorter ones (the pauses we hear end
        # are the short ones; long ones usually got an answer, so learning down would cut fluent callers off)
        mixed = [[max(a, (self.PRIOR * a + n * b) / (self.PRIOR + n)) for a, b in zip(row, emp)] for row in self.S]
        return LearnedWait(self.beta, self.edges, self.grid, mixed, self.theta)._wait(p_finished, p_still)

    PRIOR = 10
