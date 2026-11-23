"""Play Twilio against the running agent: stream the caller side of a recorded call over a real media stream
and check the agent keeps answering (no dead air, no broken voice). Costs a few cents of Deepgram + Haiku.

  python demos/phone/simulate_call.py demos/phone/recordings/<call>.wav [public base url]
  NOISE=1 python ...   adds line hiss and a short noise burst every 3.5 s (breath, rustle, cough)

The base URL defaults to the address start.py printed in data/phone_bot.log. The conversation itself won't make
sense (the caller side is fixed); this tests the plumbing: tokens, audio in and out, replies after each turn.
"""
import asyncio
import audioop
import base64
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import wave

import websockets
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
load_dotenv(os.path.join(ROOT, ".env"))


def twiml_token(base: str) -> str:
    url, params = f"{base}/twiml", {"CallSid": "CAsimulated", "From": "+10000000000", "To": "+10000000001"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if os.getenv("TWILIO_AUTH_TOKEN"):
        from twilio.request_validator import RequestValidator
        headers["X-Twilio-Signature"] = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"]).compute_signature(url, params)
    body = urllib.request.urlopen(urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(),
                                                         headers=headers, method="POST"), timeout=15).read().decode()
    return re.search(r'Stream url="([^"]+)"', body).group(1)


async def main(wav_path: str, base: str):
    with wave.open(wav_path) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    if ch == 2:
        pcm = audioop.tomono(pcm, 2, 1, 0)  # caller is the left channel
    if sr != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, sr, 8000, None)
    if os.getenv("NOISE"):  # real lines: steady hiss plus a short burst (breath, rustle, cough) every 3.5 s
        import numpy as np
        x = np.frombuffer(pcm, np.int16).astype(float)
        rng = np.random.default_rng(0)
        x += rng.normal(0, 150, len(x))
        for start in range(8000, len(x) - 4000, 28000):
            n = int(rng.uniform(0.2, 0.6) * 8000)
            x[start:start + n] += rng.normal(0, 2500, n) * np.hanning(n)
        pcm = np.clip(x, -32768, 32767).astype(np.int16).tobytes()
    chunks = [audioop.lin2ulaw(pcm[i:i + 320], 2) for i in range(0, len(pcm), 320)]  # 20 ms frames

    ws_url = twiml_token(base)
    agent_audio = []  # (t, seconds of audio received)
    async with websockets.connect(ws_url) as ws:
        sid = "MZsimulated"
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "streamSid": sid, "start": {
            "streamSid": sid, "callSid": "CAsimulated", "accountSid": "ACsimulated", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))
        t0 = time.time()

        async def receive():
            async for msg in ws:
                m = json.loads(msg)
                if m.get("event") == "media":
                    agent_audio.append((time.time() - t0, len(base64.b64decode(m["media"]["payload"])) / 8000))

        rx = asyncio.create_task(receive())
        for i, c in enumerate(chunks):
            await ws.send(json.dumps({"event": "media", "streamSid": sid,
                                      "media": {"payload": base64.b64encode(c).decode()}}))
            await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.02 - time.time()))
        await asyncio.sleep(10)  # let the last reply play
        await ws.send(json.dumps({"event": "stop", "streamSid": sid}))
        rx.cancel()

    # agent speaking intervals from the outbound media timing
    spans, cur = [], None
    for t, d in agent_audio:
        if cur and t - cur[1] < 1.0:
            cur[1] = t + d
        else:
            if cur:
                spans.append(cur)
            cur = [t, t + d]
    if cur:
        spans.append(cur)
    total = sum(e - s for s, e in spans)
    gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
    print(f"call {len(chunks) * 0.02:.0f}s of caller audio; agent spoke {len(spans)} times, {total:.0f}s in total")
    print(f"longest silence between agent replies: {max(gaps, default=0):.1f}s")
    return spans


if __name__ == "__main__":
    log = open(os.path.join(ROOT, "data", "phone_bot.log"), encoding="utf8", errors="replace").read()
    default = (re.findall(r"public at (https://\S+?);", log) or [None])[-1]
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else default))
