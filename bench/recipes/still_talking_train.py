"""Train the 'still talking' model on SEP-28k (+ FluencyBank SEP labels) and FluencyBank turn structure.

  python bench/recipes/still_talking_train.py data      # needs torch; run fluencybank.py first

Two tasks, one small CNN+GRU (~0.3M params) on 3 s log-mel windows ending where speech stops:
  cont   FluencyBank: the pause after this speech run is mid-turn (1) or a turn end (0)
  kinds  SEP-28k: block / prolongation / sound rep / word rep / interjection in the clip (>= 2 of 3 raters)
5-fold cross-validation by interview: every FluencyBank gap is scored by a model that never heard that
interview. Writes
  data/still_talking/signals_cv.json   {interview: {gap index: {p_still_talking, still_talking_kind}}} for the bench
  data/still_talking/still_talking.onnx   model trained on everything (stays under data/: dataset licences)
"""
import csv
import json
import os
import re
import sys
import wave

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from turnheed.still_talking import HEADS, KINDS, SR, WIN, log_mel  # noqa: E402

FOLDS, EPOCHS, BATCH = 5, 12, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"


def read_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def inv_starts(cha):
    return [int(m.group(1)) / 1000 for line in open(cha, encoding="utf8") if line.startswith("*INV:")
            for m in [re.search(r"\x15(\d+)_(\d+)\x15", line)] if m]


def kind_labels(row):
    return [float(int(row[h]) >= 2) for h in HEADS]


def build(root):
    """-> mel (N,298,64) f16, cont (N,) {1,0,-1 none}, kinds (N,5) {1,0,-1 none}, iid (N,) '' for podcasts, gap (N,)"""
    mel, cont, kinds, iid, gap = [], [], [], [], []

    def add(x, c, k, i="", g=-1):
        mel.append(log_mel(x).astype(np.float16)), cont.append(c), kinds.append(k), iid.append(i), gap.append(g)

    sep = f"{root}/sep28k-repo"
    for r in csv.DictReader(open(f"{sep}/SEP-28k_labels.csv", encoding="utf8"), skipinitialspace=True):
        p = f"{root}/sep28k/clips/{r['Show']}/{r['EpId']}/{r['Show']}_{r['EpId']}_{r['ClipId']}.wav"
        if os.path.exists(p):
            add(read_wav(p), -1, kind_labels(r))
    print("sep28k clips", len(mel), flush=True)

    fb = f"{root}/fluencybank"
    ep2iid = {r[1].strip(): os.path.basename(r[2].strip())[:-4]
              for r in csv.reader(open(f"{sep}/fluencybank_episodes.csv", encoding="utf8")) if r}
    audio = {}

    def get(i):  # one interview in memory at a time (~100 MB each as float32)
        if i not in audio:
            audio.clear()
            audio[i] = read_wav(f"{fb}/wav16k/{i}.wav")
        return audio[i]

    for r in csv.DictReader(open(f"{sep}/fluencybank_labels.csv", encoding="utf8"), skipinitialspace=True):
        i = ep2iid.get(r["EpId"])
        if i and os.path.exists(f"{fb}/wav16k/{i}.wav"):
            add(get(i)[int(r["Start"]):int(r["Stop"])], -1, kind_labels(r), i)
    print("+ fluencybank sep clips", len(mel), flush=True)

    for rec in json.load(open(f"{fb}/bench.json", encoding="utf8")):
        x, inv = get(rec["id"]), inv_starts(f"{fb}/Voices-AWS/interview/{rec['id']}.cha")
        for g, s in enumerate(rec["segments"]):
            end = int(s["end"] * SR)
            eot = bool(s.get("end_of_turn"))
            # the interview is one mixed track: if the interviewer already started, the window hears them.
            # keep those for scoring (gap index), never for training (cont = -1)
            overlap = eot and any(s["end"] - 0.5 <= t < s["end"] for t in inv)
            add(x[max(0, end - WIN):end], -1 if overlap else float(not eot), [-1.0] * 5, rec["id"], g)
    print("+ fluencybank gaps", len(mel), flush=True)
    return np.stack(mel), np.array(cont, np.float32), np.array(kinds, np.float32), np.array(iid), np.array(gap)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o, s: [nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU()]
        self.conv = nn.Sequential(*c(1, 32, (2, 2)), *c(32, 64, (1, 2)), *c(64, 64, (1, 2)))  # 64 mel -> 8
        self.gru = nn.GRU(64 * 8, 128, batch_first=True)
        self.cont, self.kinds = nn.Linear(128, 1), nn.Linear(128, len(HEADS))

    def forward(self, mel):                       # (B, 298, 64)
        h = self.conv(mel[:, None])               # (B, 64, 149, 8)
        h = h.permute(0, 2, 1, 3).flatten(2)      # (B, 149, 512)
        h = self.gru(h)[0][:, -1]                 # causal: state at the moment speech stopped
        return self.cont(h)[:, 0], self.kinds(h)  # logits


def train(X, C, K, idx, seed=0):
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(idx) // BATCH + 1))
    c = C[idx][C[idx] >= 0]
    w_pos, w_neg = len(c) / (2 * c.sum()), len(c) / (2 * (len(c) - c.sum()))  # balanced: 0.5 is prior-free
    bce = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(idx), len(idx) // BATCH + 1):
            x = torch.from_numpy(X[b].astype(np.float32)).to(dev)
            x = x * (torch.rand(len(b), 1, 1, device=dev) * 0.4 + 0.8)  # gain jitter on normalised log-mel
            cy, ky = torch.from_numpy(C[b]).to(dev), torch.from_numpy(K[b]).to(dev)
            cl, kl = net(x)
            cm, km = (cy >= 0).float(), (ky >= 0).float()
            cw = cm * torch.where(cy > 0.5, w_pos, w_neg)
            loss = (bce(cl, cy.clamp(min=0)) * cw).sum() / cm.sum().clamp(min=1) \
                + (bce(kl, ky.clamp(min=0)) * km).sum() / km.sum().clamp(min=1)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, X, idx):
    out_c, out_k = [], []
    for b in np.array_split(idx, max(1, len(idx) // 512)):
        cl, kl = net(torch.from_numpy(X[b].astype(np.float32)).to(dev))
        out_c.append(torch.sigmoid(cl).cpu().numpy()), out_k.append(torch.sigmoid(kl).cpu().numpy())
    return np.concatenate(out_c), np.concatenate(out_k)


def auc(p, y):
    r = np.argsort(np.argsort(p)) + 1
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main(root):
    out = f"{root}/still_talking"
    os.makedirs(out, exist_ok=True)
    cache = f"{out}/features.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        X, C, K, I, G = d["X"], d["C"], d["K"], d["I"], d["G"]
    else:
        X, C, K, I, G = build(root)
        np.savez(cache, X=X, C=C, K=K, I=I, G=G)

    ids = sorted(set(I) - {""})
    fold_of = {i: n % FOLDS for n, i in enumerate(np.random.default_rng(0).permutation(ids))}
    signals = {}
    for f in range(FOLDS):
        test = np.array([i != "" and fold_of[i] == f for i in I])
        net = train(X, C, K, np.where(~test)[0], seed=f)
        gaps = np.where(test & (G >= 0))[0]
        pc, pk = predict(net, X, gaps)
        y = C[gaps]
        print(f"fold {f}: {len(set(I[gaps]))} interviews, gap AUC {auc(pc[y >= 0], y[y >= 0]):.3f}", flush=True)
        for j, p, k in zip(gaps, pc, pk):
            signals.setdefault(str(I[j]), {})[str(int(G[j]))] = {
                "p_still_talking": round(float(p), 4), "still_talking_kind": KINDS[int(np.argmax(k))]}
    json.dump(signals, open(f"{out}/signals_cv.json", "w"))

    net = train(X, C, K, np.arange(len(X)), seed=99)
    class Probs(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, mel):
            c, k = self.n(mel)
            return torch.sigmoid(c), torch.sigmoid(k)

    torch.onnx.export(Probs(net.cpu()), torch.zeros(1, X.shape[1], X.shape[2]), f"{out}/still_talking.onnx",
                      input_names=["mel"], output_names=["cont", "kinds"],
                      dynamic_axes={"mel": {0: "b"}, "cont": {0: "b"}, "kinds": {0: "b"}})
    print("params", sum(p.numel() for p in net.parameters()))


if __name__ == "__main__":
    main(sys.argv[1])
