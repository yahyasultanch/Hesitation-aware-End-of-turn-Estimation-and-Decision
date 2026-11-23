"""Learn how long to wait from the evidence, instead of hand-tuning it ("wait only when it matters").

  python bench/recipes/fit_wait.py data [--phone]   # --phone: fit on phone-line signals -> wait_model_phone.json

For every pause: pi = P(the turn has ended | p_finished, p_still_talking) from a 3-weight logistic fit, and the
chance an in-turn pause of that evidence bin lasts longer than s seconds, S(s | bin), from the data. After s
seconds of silence the posterior that the turn ended is  pi / (pi + (1 - pi) * S(s)).  The agent answers at the
first s where that reaches theta: fast when the evidence says finished, longer only where real mid-turn pauses
of that kind run long. theta is the single knob; it is chosen to minimise cut-offs + late replies, averaged
over stuttered and fluent speakers, so neither failure is traded away for the other.

Cross-validated by speaker (5 folds per corpus). Evaluation is exact for a stateless policy: a reply at w cuts
an in-turn pause off if the pause is longer than w; at a turn end it is late if w > 2 s or the talker resumes
first. Session adaptation (patience) is not in this score; the full bench checks it afterwards.
Writes data/still_talking/wait_model.json (fit on everything) and prints the comparison.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from turnheed.policy import CONTEXT_PATIENCE  # noqa: E402,F401  (the wait model composes with it)

GRID = np.round(np.arange(0.2, 6.01, 0.05), 2)  # candidate waits, s (0.2 = when VAD/Smart Turn report)
BINS = 5
LATE = 2.0


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load(root, channel=""):
    rows = []
    for c in ("fluencybank", "buckeye"):
        recs = json.load(open(f"{root}/{c}/bench.json", encoding="utf8"))
        st = json.load(open(f"{root}/{c}/smart_turn{channel}_signals.json", encoding="utf8"))
        v2 = json.load(open(f"{root}/{c}/still_talking_v2{channel}_signals.json", encoding="utf8"))
        for r in recs:
            S = r["segments"]
            for g, s in enumerate(S):
                gap = S[g + 1]["start"] - s["end"] if g + 1 < len(S) else 99.0
                rows.append((c, r["id"] if c == "fluencybank" else r["id"][:3], bool(s.get("end_of_turn")), gap,
                             st[r["id"]][str(g)]["p_finished"], v2[r["id"]][str(g)]["p_still_talking"]))
    corpus, spk, end, gap, pf, ps = map(np.array, zip(*rows))
    X = np.stack([logit(pf), logit(ps), np.ones(len(pf))], 1)
    return corpus, spk, end.astype(float), gap.astype(float), X


def fit(X, y, w, iters=500, lr=0.5):
    """Weighted logistic regression, plain gradient descent (3 weights)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        b -= lr * (X.T @ (w * (p - y))) / w.sum()
    return b


def survival(score, gap, end, edges):
    """S[bin, k] = P(in-turn pause > GRID[k] | evidence bin)."""
    b = np.clip(np.searchsorted(edges, score), 0, BINS - 1)
    S = np.ones((BINS, len(GRID)))
    for k in range(BINS):
        g = gap[(b == k) & (end == 0)]
        if len(g):
            S[k] = (g[:, None] > GRID[None, :]).mean(0)
    return S


class WaitModel:
    def __init__(self, beta, edges, S):
        self.beta, self.edges, self.S = np.asarray(beta), np.asarray(edges), np.asarray(S)

    def pi(self, X):
        return 1 / (1 + np.exp(-X @ self.beta))

    def wait(self, X, theta):
        pi = self.pi(X)
        b = np.clip(np.searchsorted(self.edges, X @ self.beta), 0, BINS - 1)
        S = np.maximum(self.S[b], 1e-6)
        post = pi[:, None] / (pi[:, None] + (1 - pi[:, None]) * S)
        ok = post >= theta
        return np.where(ok.any(1), GRID[ok.argmax(1)], GRID[-1])

    def to_json(self):
        return {"beta": self.beta.tolist(), "edges": self.edges.tolist(), "grid": GRID.tolist(), "S": self.S.tolist()}


def train(X, y, gap, corpus):
    w = np.zeros(len(y))
    for c in set(corpus):  # every (corpus, class) cell weighs the same, then restore the pooled end prior
        for v in (0, 1):
            m = (corpus == c) & (y == v)
            w[m] = 1 / max(m.sum(), 1)
    beta = fit(X, y, w)
    beta[2] += np.log(y.mean() / (1 - y.mean()))  # balanced fit -> pooled prior odds of a turn end
    score = X @ beta
    edges = np.quantile(score, np.linspace(0, 1, BINS + 1)[1:-1])
    return WaitModel(beta, edges, survival(score, gap, y, edges))


def score(w, end, gap, corpus):
    out = {}
    for c, name in (("fluencybank", "stuttered"), ("buckeye", "fluent")):
        m = corpus == c
        pause, eot = m & (end == 0), m & (end == 1)
        cut = (gap[pause] > w[pause]).mean()
        late = ((w[eot] > LATE) | (gap[eot] <= w[eot])).mean()
        med = np.median(w[eot & (gap > w)]) if (eot & (gap > w)).any() else np.nan
        out[name] = (cut, late, med)
    return out


def show(name, res):
    s, f = res["stuttered"], res["fluent"]
    J = (s[0] + s[1] + f[0] + f[1]) / 2
    print(f"{name:42} stut cut {s[0]*100:5.1f}% late {s[1]*100:5.1f}% med {s[2]*1000:5.0f}ms | "
          f"fluent cut {f[0]*100:5.1f}% late {f[1]*100:5.1f}% med {f[2]*1000:5.0f}ms | cut+late {J*100:5.1f}")
    return J


def main(root, channel=""):
    corpus, spk, end, gap, X = load(root, channel)
    print(f"signals: {'phone line (8 kHz mu-law)' if channel else 'clean 16 kHz'}")
    fold = np.zeros(len(end), int)
    for c in set(corpus):
        groups = sorted(set(spk[corpus == c]))
        f_of = {g: k % 5 for k, g in enumerate(np.random.default_rng(0).permutation(groups))}
        fold[corpus == c] = [f_of[g] for g in spk[corpus == c]]

    thetas = np.round(np.arange(0.05, 0.96, 0.05), 2)
    W = {t: np.zeros(len(end)) for t in thetas}
    for f in range(5):
        tr, te = fold != f, fold == f
        m = train(X[tr], end[tr], gap[tr], corpus[tr])
        for t in thetas:
            W[t][te] = m.wait(X[te], t)

    pf, ps = 1 / (1 + np.exp(-X[:, 0])), 1 / (1 + np.exp(-X[:, 1]))
    print("cross-validated by speaker; stateless (no session patience)\n")
    show("baseline: Smart Turn, Pipecat default", score(np.where(pf >= 0.5, 0.2, 3.0), end, gap, corpus))
    for name, T0, tau, hold in (("layer formula (0.5s, tau .6, hold 3)", 0.5, 0.6, 3.0),
                                ("layer formula, demo now (0.8s, hold 6)", 0.8, 0.6, 6.0)):
        z = (logit(ps) + logit(1 - pf)) / 2
        show(name, score(np.clip(T0 + tau * z, 0.2, hold), end, gap, corpus))
    for t in (500, 800, 1200):
        show(f"timeout {t} ms", score(np.full(len(end), t / 1000), end, gap, corpus))
    print()
    best = min(thetas, key=lambda t: show(f"learned wait, theta {t:.2f}", score(W[t], end, gap, corpus)))
    print(f"\nbest theta {best:.2f}:")
    show(f"learned wait, theta {best:.2f}", score(W[best], end, gap, corpus))

    model = train(X, end, gap, corpus)
    out = {**model.to_json(), "theta": float(best)}
    json.dump(out, open(f"{root}/still_talking/wait_model{channel}.json", "w"))
    print("beta (p_finished, p_still, bias):", np.round(model.beta, 3).tolist())


if __name__ == "__main__":
    main(sys.argv[1], "_phone" if "--phone" in sys.argv else "")
