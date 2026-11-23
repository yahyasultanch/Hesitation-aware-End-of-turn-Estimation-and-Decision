"""Still-talking v2: a small head on Smart Turn's pooled encoder output, trained on FluencyBank (stuttered)
and Buckeye (fluent) turn structure. Run bench/recipes/smart_turn_embed.py first.

  python bench/recipes/still_talking_v2_train.py data

5-fold cross-validation by speaker within each corpus (FluencyBank interview, Buckeye talker sXX); every
(corpus, class) cell weighs the same in the loss, so neither corpus nor the rare turn ends dominate and
0.5 stays prior-free. Turn ends where the interviewer already started are scored, never trained on.
Trains on clean audio and, when smart_turn_embed.py --phone has run, the same gaps through a simulated phone
line (8 kHz mu-law), scored separately. Writes data/<set>/still_talking_v2[_phone]_signals.json,
data/<set>/smart_turn_phone_signals.json and data/still_talking/still_talking_v2_head.onnx.
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from still_talking_train import auc, inv_starts  # noqa: E402

FOLDS, EPOCHS = 5, 60


def overlaps(root, name, recs):
    """(id, gap) of turn ends whose 3 s before speech end already contain the interviewer starting."""
    out = set()
    for rec in recs:
        if name == "fluencybank":
            inv = inv_starts(f"{root}/Voices-AWS/interview/{rec['id']}.cha")
        else:
            continue  # Buckeye interviewer was not miked; <IVER_overlap> spans are rare and quiet
        for g, s in enumerate(rec["segments"]):
            if s.get("end_of_turn") and any(s["end"] - 0.5 <= t < s["end"] for t in inv):
                out.add((rec["id"], g))
    return out


def load(root, name, channel=""):
    d = np.load(f"{root}/{name}/smart_turn_embed{channel}.npz")
    recs = json.load(open(f"{root}/{name}/bench.json", encoding="utf8"))
    eot = {(r["id"], g): bool(s.get("end_of_turn")) for r in recs for g, s in enumerate(r["segments"])}
    ov = overlaps(f"{root}/{name}", name, recs)
    y = np.array([0.0 if eot[(i, int(g))] else 1.0 for i, g in zip(d["id"], d["gap"])], np.float32)
    train_ok = np.array([(i, int(g)) not in ov for i, g in zip(d["id"], d["gap"])])
    spk = np.array([i if name == "fluencybank" else i[:3] for i in d["id"]])
    return d["emb"].astype(np.float32), y, train_ok, spk, d["id"], d["gap"]


class Head(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.mu, self.sd = nn.Parameter(torch.tensor(mu), False), nn.Parameter(torch.tensor(sd), False)
        self.net = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, e):
        return torch.sigmoid(self.net((e - self.mu) / self.sd))[:, 0]


def train(X, y, w, seed):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    net = Head(mu, sd)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    X, y, w = map(torch.from_numpy, (X, y, w))
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(X)), len(X) // 512 + 1):
            p = net(X[b]).clamp(1e-5, 1 - 1e-5)
            loss = -(w[b] * (y[b] * p.log() + (1 - y[b]) * (1 - p).log())).sum() / w[b].sum()
            opt.zero_grad(), loss.backward(), opt.step()
    return net.eval()


def data(root):
    """All training rows (clean + phone channels) with weights and the speaker folds used for cross-validation."""
    # clean 16 kHz audio, plus the same audio through a simulated phone line when it has been extracted
    channels = [c for c in ("", "_phone") if os.path.exists(f"{root}/fluencybank/smart_turn_embed{c}.npz")]
    parts = [(c, n, load(root, n, c)) for c in channels for n in ("fluencybank", "buckeye")]
    X = np.concatenate([d[0] for _, _, d in parts])
    y = np.concatenate([d[1] for _, _, d in parts])
    ok = np.concatenate([d[2] for _, _, d in parts])
    chan = np.concatenate([[c] * len(d[0]) for c, _, d in parts])
    corpus = np.concatenate([[n] * len(d[0]) for _, n, d in parts])
    spk = np.concatenate([d[3] for _, _, d in parts])
    ids = np.concatenate([d[4] for _, _, d in parts])
    gaps = np.concatenate([d[5] for _, _, d in parts])
    names = ("fluencybank", "buckeye")

    w = np.zeros(len(y), np.float32)
    for c in names:
        for v in (0.0, 1.0):
            m = (corpus == c) & (y == v) & ok
            w[m] = 1.0 / m.sum()
    fold = np.zeros(len(y), int)
    for c in names:  # same speaker -> same fold in every channel
        groups = sorted(set(spk[corpus == c]))
        f_of = {g: k % FOLDS for k, g in enumerate(np.random.default_rng(0).permutation(groups))}
        fold[corpus == c] = [f_of[g] for g in spk[corpus == c]]
    return X, y, ok, chan, corpus, spk, ids, gaps, w, fold, channels, names


def fold_nets(root):
    """The 5 cross-validation heads (same seeds as main) and {(corpus, speaker): fold}, so new audio of a speaker
    is scored by the head that never saw them (e.g. bench/recipes/noise_bench.py)."""
    X, y, ok, chan, corpus, spk, ids, gaps, w, fold, _, _ = data(root)
    nets = [train(X[(fold != f) & ok], y[(fold != f) & ok], w[(fold != f) & ok], seed=f) for f in range(FOLDS)]
    return nets, {(c, s): f for c, s, f in zip(corpus, spk, fold)}


def main(root):
    X, y, ok, chan, corpus, spk, ids, gaps, w, fold, channels, names = data(root)
    p = np.zeros(len(y), np.float32)
    for f in range(FOLDS):
        tr, te = (fold != f) & ok, fold == f
        net = train(X[tr], y[tr], w[tr], seed=f)
        with torch.no_grad():
            p[te] = net(torch.from_numpy(X[te])).numpy()
        print(f"fold {f}: " + "  ".join(f"{c}{ch or ' clean'} AUC {auc(p[te & (corpus == c) & (chan == ch)], y[te & (corpus == c) & (chan == ch)]):.3f}"
                                        for ch in channels for c in names), flush=True)
    for ch in channels:
        for c in names:
            m = (corpus == c) & (chan == ch)
            print(f"{c}{ch or ' clean'}: cross-validated AUC {auc(p[m], y[m]):.3f}")
            sig = {}
            for i, g, q in zip(ids[m], gaps[m], p[m]):
                sig.setdefault(str(i), {})[str(int(g))] = {"p_still_talking": round(float(q), 4)}
            json.dump(sig, open(f"{root}/{c}/still_talking_v2{ch}_signals.json", "w"))
            if ch:  # Smart Turn's own reading of the phone audio, for the bench and the wait fit
                d = np.load(f"{root}/{c}/smart_turn_embed{ch}.npz")
                st = {}
                for i, g, q in zip(d["id"], d["gap"], d["p_finished"]):
                    st.setdefault(str(i), {})[str(int(g))] = {"p_finished": round(float(q), 4)}
                json.dump(st, open(f"{root}/{c}/smart_turn{ch}_signals.json", "w"))

    net = train(X[ok], y[ok], w[ok], seed=99)
    torch.onnx.export(net, torch.zeros(1, 384), f"{root}/still_talking/still_talking_v2_head.onnx",
                      input_names=["emb"], output_names=["p_still"], dynamic_axes={"emb": {0: "b"}, "p_still": {0: "b"}})


if __name__ == "__main__":
    main(sys.argv[1])
