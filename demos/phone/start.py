"""One command for the phone demo: tunnel + agent + Twilio webhook, kept in step.

  python demos/phone/start.py

Starts a tunnel (ngrok if installed, else a cloudflared quick tunnel; TUNNEL= to choose), reads its public address, starts the agent with it (PUBLIC_HOST, used to check
Twilio's request signatures) and points the Twilio number's voice webhook at it through the Twilio API. Every
minute it checks the public address; quick tunnels can drop (e.g. after the laptop sleeps), so on failure it
starts a new tunnel and updates Twilio again. Ctrl+C stops everything.

NGROK_DOMAIN: your free static ngrok domain, so the address never changes.
Needs in .env: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN (without them the webhook must be set by hand; the
address is printed). TWILIO_NUMBER picks the number if the account has several. CLOUDFLARED overrides the path.
Extra environment (RECORD, TRANSCRIPTS, DEMO_MODE...) is passed to the agent unchanged.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
load_dotenv(os.path.join(HERE, ".env"))
load_dotenv(os.path.join(ROOT, ".env"))
PORT = os.getenv("PORT", "8765")
CLOUDFLARED = os.getenv("CLOUDFLARED") or shutil.which("cloudflared") or "cloudflared"


def start_tunnel():
    if os.getenv("TUNNEL", "ngrok" if shutil.which("ngrok") else "cloudflared") == "ngrok":
        return start_ngrok()
    return start_cloudflared()


def start_ngrok():
    """ngrok: steadier than a Cloudflare quick tunnel for long calls; with NGROK_DOMAIN (a free static domain from
    the ngrok dashboard) the address never changes, so the Twilio webhook never needs updating."""
    cmd = ["ngrok", "http", PORT, "--log=stdout"]
    if os.getenv("NGROK_DOMAIN"):
        cmd += ["--url", os.environ["NGROK_DOMAIN"]]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    url, deadline = None, time.time() + 30
    while time.time() < deadline and url is None:
        line = p.stdout.readline()
        m = re.search(r"url=(https://\S+)", line)
        url = m.group(1) if m else None
    if url is None:
        p.kill()
        raise RuntimeError("ngrok did not report a public address within 30 s")
    _keep_log(p)
    return p, url


def _keep_log(p):
    log = open(os.path.join(ROOT, "data", "tunnel.log"), "a", encoding="utf8")

    def drain():  # keep the tunnel's own log: a dropped call can be matched to a tunnel reconnect
        for line in p.stdout:
            log.write(line)
            log.flush()
    threading.Thread(target=drain, daemon=True).start()


def start_cloudflared():
    p = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    url, deadline = None, time.time() + 60
    while time.time() < deadline and url is None:
        line = p.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        url = m.group(0) if m else None
    if url is None:
        p.kill()
        raise RuntimeError("cloudflared did not report a public address within 60 s")
    _keep_log(p)
    return p, url


def point_twilio(url):
    sid, token = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and token):
        print(f"\n  Set the Twilio number's voice webhook by hand: {url}/twiml  (HTTP POST)\n"
              "  (add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to .env to have this done automatically)\n", flush=True)
        return
    from twilio.rest import Client
    numbers = Client(sid, token).incoming_phone_numbers.list(
        phone_number=os.getenv("TWILIO_NUMBER") or os.getenv("TWILIO_PHONE_NUMBER") or None)
    if len(numbers) != 1:
        print(f"  {len(numbers)} Twilio numbers match; set TWILIO_NUMBER to choose one", flush=True)
        return
    numbers[0].update(voice_url=f"{url}/twiml", voice_method="POST")
    print(f"  Twilio {numbers[0].phone_number} -> {url}/twiml", flush=True)


def healthy(url) -> bool:
    try:
        return urllib.request.urlopen(f"{url}/health", timeout=10).status == 200
    except Exception:
        return False


def main():
    tunnel, url = start_tunnel()
    env = dict(os.environ, PUBLIC_HOST=url.removeprefix("https://"))
    if os.getenv("NGROK_DOMAIN") and os.getenv("TWILIO_ACCOUNT_SID"):
        print("  static ngrok domain: the Twilio webhook stays the same across restarts", flush=True)
    agent = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "bot.py")], env=env)
    print(f"  agent on 127.0.0.1:{PORT}, public at {url}; staff view http://localhost:{PORT}/bookings", flush=True)
    point_twilio(url)
    fails = 0
    try:
        while agent.poll() is None:
            time.sleep(60)
            fails = 0 if healthy(url) else fails + 1
            if fails >= 3 or tunnel.poll() is not None:
                print("  tunnel lost: starting a new one", flush=True)
                tunnel.kill()
                tunnel, url = start_tunnel()
                agent.kill()
                agent.wait()
                env["PUBLIC_HOST"] = url.removeprefix("https://")
                agent = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "bot.py")], env=env)
                point_twilio(url)
                fails = 0
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.kill()
        agent.kill()


if __name__ == "__main__":
    main()
