"""Measure the words' veto on real phone calls: Haiku judges each SpokenWOZ pause complete or mid-sentence from
the caller's words so far (same completion instructions as the demo), and the audio policy's turn ends are
vetoed when it says mid-sentence.

  python bench/recipes/completion_veto_eval.py data/spokenwoz|data/fluencybank [n_pauses] [n_turn_ends]
  (~$0.25 per 1,000 gaps; judgements cached per set)
  JUDGE=ollama:<model> python ...   same check with a local open model (free; cached per model)

Sample: every in-turn pause + n_turn_ends random turn ends (default = as many as the pauses). Judgements are
cached in <root>/completion_judgements.json, so re-runs cost nothing.
A pause is cut off when the policy answers inside it AND Haiku says complete; a turn end is late when the
policy waits > 2 s, the caller resumes first, or Haiku wrongly says mid-sentence (the agent would then wait
for its 8 s check-in).
"""
import asyncio
import json
import os
import random
import sys

import anthropic
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "demos", "phone"))
from bot import COMPLETION, HAIKU  # noqa: E402
from turnheed.wait import LearnedWait  # noqa: E402

SYSTEM = ("You are an AI phone assistant that books hotels, restaurants, trains, taxis and attractions. "
          "Keep replies to one short sentence.\n" + COMPLETION.completion_instructions)


def turn_texts(root):
    """Per gap: (agent's last line, caller words of this turn so far, is turn end, gap length)."""
    data = json.load(open(f"{root}/raw/data.json", encoding="utf8"))
    recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}
    out = {}
    for did, rec in recs.items():
        log = data[did]["log"]
        words = sorted((w["EndTime"] / 1000, w["Word"], t["tag"]) for t in log for w in t["words"])
        segs, so_far, last_agent, i, agent_buf = rec["segments"], [], "", 0, []
        for g, s in enumerate(segs):
            while i < len(words) and words[i][0] <= s["end"] + 0.05:
                t, w, tag = words[i]
                if tag == "user":
                    if agent_buf:
                        last_agent, agent_buf = " ".join(agent_buf), []
                    so_far.append(w)
                else:
                    agent_buf.append(w)
                i += 1
            gap = segs[g + 1]["start"] - s["end"] if g + 1 < len(segs) else 99.0
            out[(did, g)] = (last_agent, " ".join(so_far), bool(s.get("end_of_turn")), gap)
            if s.get("end_of_turn"):
                so_far = []
    return out


def turn_texts_fluencybank(root):
    """Same as turn_texts for FluencyBank interviews (people who stutter): caller words from the Timestamped
    CSVs (fillers dropped, as Deepgram drops them; repetitions kept), the interviewer's last line from CHAT."""
    import csv
    import re
    recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}
    words = {}
    for r in csv.DictReader(open(f"{root}/TimeStamped/metadata.csv", encoding="utf8")):
        iid, off = r["segid"].rsplit("_", 1)[0], float(r["segstart"]) - 0.05
        for w in csv.DictReader(open(f"{root}/TimeStamped/csvs/{r['segid']}.csv", encoding="utf8")):
            if w["fp"] != "1":
                words.setdefault(iid, []).append((off + float(w["wordend"]), w["word"]))
    out = {}
    for iid, rec in recs.items():
        inv = []
        for line in open(f"{root}/Voices-AWS/interview/{iid}.cha", encoding="utf8"):
            m = re.search(r"\x15(\d+)_(\d+)\x15", line)
            if line.startswith("*INV:") and m:
                text = re.sub(r"\x15.*?\x15|\[.*?\]|&[-=+]?\S+|[+<>/()]|\bxxx\b", " ", line[5:])
                inv.append((int(m.group(1)) / 1000, " ".join(text.split())))
        ws, i, so_far = sorted(words.get(iid, [])), 0, []
        segs = rec["segments"]
        for g, s in enumerate(segs):
            while i < len(ws) and ws[i][0] <= s["end"] + 0.05:
                so_far.append(ws[i][1])
                i += 1
            last_inv = next((t for st, t in reversed(inv) if st < s["end"]), "")
            gap = segs[g + 1]["start"] - s["end"] if g + 1 < len(segs) else 99.0
            out[(iid, g)] = (last_inv, " ".join(so_far), bool(s.get("end_of_turn")), gap)
            if s.get("end_of_turn"):
                so_far = []
    return out


JUDGE = os.getenv("JUDGE", "haiku")  # haiku | ollama:<model> (local open model, e.g. ollama:qwen3:8b)


def judge_local(agent, caller):
    """Same prompt and turns, asked of a local open model through Ollama. Greedy, no thinking. Also returns
    P(complete) = P(✓) / P(✓ or ○ or ◐) for the first token, so the cut-point can be tuned like any score."""
    import urllib.error
    import urllib.request
    body = {"model": JUDGE.split(":", 1)[1], "stream": False, "think": False,
            "logprobs": True, "top_logprobs": 10, "options": {"temperature": 0, "num_predict": 4},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "(call connected)"},
                         {"role": "assistant", "content": agent or "Hello, how can I help?"},
                         {"role": "user", "content": caller or "..."}]}
    for attempt in range(6):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=120))
        except urllib.error.HTTPError:  # models without a thinking mode reject "think": retry without it
            if "think" not in body:
                raise
            body.pop("think")
            continue
        except (TimeoutError, urllib.error.URLError, ConnectionError):  # e.g. the PC slept, Ollama reloading
            import time
            time.sleep(10 * (attempt + 1))
            continue
        text = r["message"]["content"].strip()
        mark = next((c for c in text if c in "✓○◐"), "?")  # weaker models sometimes lead with a word
        top = (r.get("logprobs") or [{}])[0].get("top_logprobs", [])
        pr = {m: sum(np.exp(t["logprob"]) for t in top if t["token"].strip() == m) for m in "✓○◐"}
        p = float(pr["✓"] / sum(pr.values())) if sum(pr.values()) > 0 else None
        return [mark, p], r.get("prompt_eval_count", 0), r.get("eval_count", 0)
    return ["?", None], 0, 0


BUCKEYE_PROMPT = "Tell me a bit about that, and about life around here."


def turn_texts_buckeye(root):
    """Same as turn_texts for Buckeye (fluent interviews): the talker's words from the hand-corrected .words files
    (fillers, noises and transcriber notes dropped, as Deepgram drops them). Buckeye does not transcribe the
    interviewer, so their line is a neutral prompt like the corpus's conversational interviews; with the default
    "how can I help?" the check waited for a request and called finished life stories mid-sentence."""
    import io
    import zipfile
    from buckeye import is_speech, parse_words
    recs = {r["id"]: r for r in json.load(open(f"{root}/bench.json", encoding="utf8"))}
    words = {}
    for n in range(1, 41):
        outer = zipfile.ZipFile(f"{root}/zips/s{n:02d}.zip")
        for name in outer.namelist():
            sid = os.path.basename(name)[:-4]
            if not name.endswith(".zip") or sid not in recs:
                continue
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            f = next(x for x in inner.namelist() if x.endswith(f"{sid}.words"))
            for a, b, label in parse_words(inner.read(f).decode("latin-1")):
                if not is_speech(label) or label.startswith("<HES") or label.lower() in ("um", "uh"):
                    continue
                if label.startswith("<CUTOFF-"):  # a broken-off word: what was actually said, before "="
                    label = label[8:].split("=")[0].rstrip(">")
                elif label.startswith("<"):
                    continue
                words.setdefault(sid, []).append((b, label))
    out = {}
    for sid, rec in recs.items():
        ws, i, so_far, segs = sorted(words.get(sid, [])), 0, [], rec["segments"]
        for g, s in enumerate(segs):
            while i < len(ws) and ws[i][0] <= s["end"] + 0.05:
                so_far.append(ws[i][1])
                i += 1
            gap = segs[g + 1]["start"] - s["end"] if g + 1 < len(segs) else 99.0
            out[(sid, g)] = (BUCKEYE_PROMPT, " ".join(so_far), bool(s.get("end_of_turn")), gap)
            if s.get("end_of_turn"):
                so_far = []
    return out


async def judge(client, sem, agent, caller):
    if JUDGE != "haiku":
        async with sem:
            return await asyncio.to_thread(judge_local, agent, caller)
    async with sem:
        for attempt in range(5):
            try:
                r = await client.messages.create(
                    model=HAIKU, max_tokens=400, output_config={"effort": "low"}, system=SYSTEM,
                    messages=[{"role": "user", "content": "(call connected)"},
                              {"role": "assistant", "content": agent or "Hello, how can I help?"},
                              {"role": "user", "content": caller or "..."}])
                text = "".join(b.text for b in r.content if b.type == "text").strip()
                if not text[:1] in ("✓", "○", "◐"):  # no marker (e.g. budget spent thinking): retry, never guess
                    continue
                return text[:1], r.usage.input_tokens, r.usage.output_tokens
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                await asyncio.sleep(2 ** attempt)
        return "?", 0, 0


async def run_judgements(items, cache_path):
    cache = json.load(open(cache_path, encoding="utf8")) if os.path.exists(cache_path) else {}
    # local models store [mark, P(complete)]; an older mark-only entry is judged again to get its score
    todo = [(k, a, c) for k, (a, c) in items.items()
            if k not in cache or (JUDGE != "haiku" and isinstance(cache[k], str))]
    client = anthropic.AsyncAnthropic() if JUDGE == "haiku" else None
    sem, tin, tout = asyncio.Semaphore(12 if JUDGE == "haiku" else int(os.getenv("JUDGE_PARALLEL", "4"))), 0, 0
    for start in range(0, len(todo), 200):
        batch = todo[start:start + 200]
        res = await asyncio.gather(*(judge(client, sem, a, c) for _, a, c in batch))
        for (k, _, _), (m, i, o) in zip(batch, res):
            cache[k] = m
            tin, tout = tin + i, tout + o
        json.dump(cache, open(cache_path, "w", encoding="utf8"))
        cost = f"  (${tin * 0.10 / 1e6 + tout * 0.50 / 1e6:.3f} so far)" if JUDGE == "haiku" else ""
        print(f"  judged {start + len(batch)}/{len(todo)}{cost}", flush=True)
    return cache


SUFFIX = "" if JUDGE == "haiku" else "_" + JUDGE.replace(":", "_").replace("/", "_")
# TAG=<name>: a separate cache for a changed prompt (e.g. TAG=v2), so earlier judgements stay comparable
SUFFIX += f"_{os.environ['TAG']}" if os.getenv("TAG") else ""


def eval_keys(root, n_pauses=None, n_ends=None):
    """The gaps this eval scores: every (or n_pauses random) in-turn pause + as many random turn ends."""
    texts = (turn_texts_fluencybank(root) if "fluencybank" in root else
             turn_texts_buckeye(root) if "buckeye" in root else turn_texts(root))
    pauses = [k for k, v in texts.items() if not v[2]]
    ends = [k for k, v in texts.items() if v[2] and v[3] < 90]
    random.Random(0).shuffle(pauses)
    random.Random(0).shuffle(ends)
    pauses = pauses[:int(n_pauses) if n_pauses else len(pauses)]
    ends = ends[:int(n_ends) if n_ends else len(pauses)]
    return texts, pauses, ends


def main(root, n_pauses=None, n_ends=None):
    stuttered = "fluencybank" in root
    texts, pauses, ends = eval_keys(root, n_pauses, n_ends)
    keys = pauses + ends
    cache = asyncio.run(run_judgements({f"{d}|{g}": texts[(d, g)][:2] for d, g in keys},
                                       f"{root}/completion_judgements{SUFFIX}.json"))
    sig = os.getenv("SIG", "")  # e.g. SIG=_snr10: the sound models' readings of the same gaps with noise mixed in
    st = json.load(open(f"{root}/smart_turn_signals{sig}.json", encoding="utf8"))
    v2 = json.load(open(f"{root}/still_talking_v2_signals{sig}.json", encoding="utf8"))
    if sig:
        print(f"sound signals: {sig}")
    eot = np.array([texts[k][2] for k in keys])
    gap = np.array([texts[k][3] for k in keys])
    pf = np.array([st[d][str(g)]["p_finished"] for d, g in keys])
    ps = np.array([v2[d][str(g)]["p_still_talking"] for d, g in keys])
    mark = {k: (v[0] if isinstance(v, list) else v) for k, v in cache.items()}
    print(f"unjudged (no marker after retries): {sum(mark[f'{d}|{g}'] == '?' for d, g in keys)}")
    complete = np.array([mark[f"{d}|{g}"] not in ("○", "◐") for d, g in keys])
    if SUFFIX and os.path.exists(f"{root}/completion_judgements.json"):
        hk = json.load(open(f"{root}/completion_judgements.json", encoding="utf8"))
        both = [k for k in (f"{d}|{g}" for d, g in keys) if k in hk]
        agree = np.mean([(hk[k] in ("○", "◐")) == (mark[k] in ("○", "◐")) for k in both])
        print(f"{JUDGE} agrees with Haiku on {agree*100:.1f}% of {len(both)} gaps (complete vs mid-sentence)")
    print(f"\n{'stuttered interviews' if stuttered else 'fluent interviews' if 'buckeye' in root else 'phone calls'}: {len(pauses)} in-turn pauses, {len(ends)} turn ends")
    print(f"{JUDGE} says mid-sentence on {np.mean(~complete[~eot])*100:.1f}% of real in-turn pauses, "
          f"and on {np.mean(~complete[eot])*100:.1f}% of labelled turn ends\n")
    RECHECK = 2.5
    BOOT = int(os.getenv("BOOT", "0"))  # BOOT=1000: speaker-level 95% intervals on cut-offs and lateness
    rec_ids = sorted({d for d, _ in keys})
    rec_idx = np.array([rec_ids.index(d) for d, _ in keys])

    def score(w, veto, done=complete, sel=None):
        """w: when the audio policy would end the turn; veto: a mid-sentence verdict holds the reply until the caller
        speaks again or the re-check. sel: subset of gaps. -> (cut-off rate, late rate, median reply, p90 reply)"""
        sel = np.ones(len(keys), bool) if sel is None else sel
        reply = np.where(done | ~veto, w, w + RECHECK)
        e, p = eot & sel, ~eot & sel
        heard = e & (gap > reply)
        return (np.mean(gap[p] > reply[p]), np.mean((reply[e] > 2.0) | (gap[e] <= reply[e])),
                np.median(reply[heard]), np.quantile(reply[heard], .9))

    def show(name, w, veto, fast=None, done=complete, sel=None):
        """fast: where the fast path answers at FAST s instead."""
        w = np.where(fast, FAST, w) if fast is not None else w
        cut, late, med, p90 = score(w, veto, done, sel)
        ci = ""
        if BOOT:  # 95% interval, resampling whole recordings (speakers / calls), not single gaps
            reply = np.where(done | ~veto, w, w + RECHECK)
            s_ = np.ones(len(keys), bool) if sel is None else sel
            c_i, l_i = (gap > reply) & ~eot & s_, ((reply > 2.0) | (gap <= reply)) & eot & s_
            p_i, e_i = ~eot & s_, eot & s_
            rng, n = np.random.default_rng(0), len(rec_ids)
            bc, bl = [], []
            for _ in range(BOOT):
                wt = np.bincount(rng.integers(0, n, n), minlength=n)[rec_idx]
                bc.append((wt * c_i).sum() / max(1, (wt * p_i).sum()))
                bl.append((wt * l_i).sum() / max(1, (wt * e_i).sum()))
            ci = (f"  [cut {np.quantile(bc, .025)*100:4.1f}-{np.quantile(bc, .975)*100:4.1f}, "
                  f"late {np.quantile(bl, .025)*100:4.1f}-{np.quantile(bl, .975)*100:4.1f}]")
        print(f"{name:52} cut-offs {cut*100:5.1f}%   late {late*100:5.1f}%   "
              f"turn-end reply median {med*1000:5.0f} ms  p90 {p90*1000:5.0f} ms{ci}")

    FAST = 0.3
    no = np.zeros(len(keys), bool)
    show("baseline: Smart Turn, Pipecat default", np.where(pf >= 0.5, 0.2, 3.0), no)
    show("timeout 1200 ms", np.full(len(keys), 1.2), no)
    wm = "data/still_talking/wait_model_phone.json" if "spokenwoz" in root else "data/still_talking/wait_model.json"
    W = {th: np.array([LearnedWait.load(wm, theta=th)(a, b) for a, b in zip(pf, ps)]) for th in (0.4, 0.5, 0.55)}
    show("layer: learned wait 0.5, audio only", W[0.5], no)
    show("layer: learned wait 0.5 + words veto", W[0.5], ~no)
    print()
    for th in (0.5, 0.55):
        for thr in (0.3, 0.5, 0.7):
            fast = complete & (ps < thr) & (pf >= 0.3)
            show(f"fast path (words done, p_still<{thr}) else learned {th}+veto", W[th], ~no, fast)

    # calibrated: tune the cut-point on P(complete) on half the recordings, report on the other half, at no more
    # lateness than Haiku's hard verdicts on the tuning half (Haiku gives no probabilities, so it stays hard)
    pc = np.array([(cache[f"{d}|{g}"][1] if isinstance(cache[f"{d}|{g}"], list) else None) for d, g in keys], float)
    if SUFFIX and not np.isnan(pc).all():
        import zlib
        pc = np.nan_to_num(pc, nan=1.0)  # no score: treat as complete, as the demo does
        half = np.array([zlib.crc32(d.encode()) % 2 == 0 for d, _ in keys])
        has_hk = os.path.exists(f"{root}/completion_judgements.json")
        if has_hk:  # no later than Haiku's hard verdicts on the tuning half
            hk = json.load(open(f"{root}/completion_judgements.json", encoding="utf8"))
            hk_done = np.array([hk.get(f"{d}|{g}", "✓") not in ("○", "◐") for d, g in keys])
            target = score(W[0.5], ~no, hk_done, half)[1]
        else:  # no Haiku reference: no later than the sound-only layer (words may only remove cut-offs)
            target = score(W[0.5], no, complete, half)[1]
            print()
            print("(no Haiku reference: cut-point tuned to stay no later than the sound-only layer)")
        res = [(score(W[0.5], ~no, pc >= t, half), t) for t in np.unique(np.quantile(pc, np.linspace(0, 1, 201)))]
        ok = [(r[0], t) for r, t in res if r[1] <= target]
        t = min(ok)[1] if ok else max(res, key=lambda x: -x[0][1])[1]
        print(f"\ncalibrated (cut-point tuned on {half.sum()} gaps, tested on the other {(~half).sum()}): "
              f"P(complete) >= {t:.3g}")
        if has_hk:
            show("held-out: learned wait 0.5 + Haiku words veto", W[0.5], ~no, done=hk_done, sel=~half)
        show("held-out: learned wait 0.5, audio only", W[0.5], no, sel=~half)
        show("held-out: timeout 1200 ms", np.full(len(keys), 1.2), no, sel=~half)
        show("held-out: learned wait 0.5 + local, calibrated", W[0.5], ~no, done=pc >= t, sel=~half)
        show("held-out: learned wait 0.5 + local, hard marker", W[0.5], ~no, sel=~half)
        print("trade-off over the whole set:")
        for t in (1e-4, 1e-3, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
            c, l, _, _ = score(W[0.5], ~no, pc >= t)
            print(f"  P(complete) >= {t:<5} cut-offs {c*100:5.1f}%   late {l*100:5.1f}%")


if __name__ == "__main__":
    main(*sys.argv[1:4])
