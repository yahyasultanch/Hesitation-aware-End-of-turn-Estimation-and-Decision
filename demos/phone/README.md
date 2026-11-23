# Phone demo: clinic booking line

A physiotherapy booking line on Twilio, in two modes with everything else identical (speech, voice, model, prompt,
database):

- `baseline`: Pipecat's default turn-taking (Smart Turn v3.2, 3 s fallback)
- `layer`: HEED: learned wait, still-talking v2, per-call pause learning, stutter-aware words check, targeted fast
  path, and a turn start that background noise cannot trigger

Speech is Deepgram (nova-3 recognition, Aura voice); the receptionist is always Claude Haiku 5.5 (about $0.005 a
call).

## Run

1. `pip install -e ".[demo]"`
2. The model files must be in `data/still_talking/` (build them with the recipes in the main README, "Reproduce").
3. Copy `.env.example` to `.env` and fill in the keys. `.env` is git-ignored; never commit it.
4. `python demos/phone/start.py`. It starts a tunnel (ngrok if installed, else a cloudflared quick tunnel), starts
   the agent, and with Twilio credentials points your number at the tunnel itself, again if the tunnel is replaced.
   Set `NGROK_DOMAIN` to a free static ngrok domain so the address does not change.
5. Call the number. Staff view (this machine only): http://localhost:8765/bookings. Switch modes between calls:
   `curl http://localhost:8765/mode/baseline` or `/mode/layer`.

## What the agent can do

Book, move and cancel appointments in the clinic database (`clinic.py`, SQLite, seeded with fictional patients),
for this week's open slots. It tells or changes a booking only when the caller's name and phone number both match;
numbers are read back from code, never written by the model, and "+44 7700 ..." and "07700 ..." are the same
caller. Every change goes to an audit table. To use a real practice-management system, implement the six methods of
`Diary` against its API.

## Checks

| Script | What it checks | Cost |
|---|---|---|
| `clinic.py` | Database rules: verification, moves, cancels, number formats (`python demos/phone/clinic.py`) | free |
| `simulate_call.py` | Plays Twilio against the running agent with a recorded caller side; `NOISE=1` adds line hiss and noise bursts | a few cents |
| `check_completion.py` | The words check stays silent on mid-sentence fragments and answers complete turns, stuttered ones included | ~1 cent |
| `check_haiku_context.py` | The model survives an interrupted reply and answers the next turn | ~1 cent |
| `replay.py` | A text replay of booking safety, including an impostor | ~1 cent |
| `replay_audio.py` | A recorded call through the layer, offline | free |

## Security

- Twilio request signatures are verified when `TWILIO_AUTH_TOKEN` is set; the media stream needs a one-time token
  issued by a valid webhook request (60 s, single use); at most `MAX_CALLS` (2) calls at once.
- The server listens on 127.0.0.1 only; the mode switch and staff view refuse any request that carries a
  proxy-forwarding header, so nothing that comes through the tunnel reaches them.
- API keys are scrubbed from every log line; transcripts are logged only with `TRANSCRIPTS=1`, calls recorded only
  with `RECORD=1` (the greeting then says so); database queries are parameterised.

## Before recording anyone

- Get written consent from everyone heard on the call. With `RECORD=1` the agent says the call is recorded.
- Recordings, logs and the database hold personal data. They stay on this machine (all git-ignored); delete them
  when you are done.
