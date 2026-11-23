# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's **Report a vulnerability** button (Security tab of this
repository), not in a public issue. Include what you found, how to reproduce it and what it affects.

## What the phone demo does to stay safe

The demo in `demos/phone/` answers real phone calls, so it is built to be run by one person on their own machine:

- **Webhook authenticity.** With `TWILIO_AUTH_TOKEN` set, every Twilio webhook must carry a valid
  `X-Twilio-Signature`; the media stream then needs a one-time token issued by that webhook (60 s, single use).
- **Exposure.** The server listens on 127.0.0.1 only and is reached from the internet through a tunnel. The staff view
  (`/bookings`) and the mode switch (`/mode/...`) refuse any request carrying a proxy-forwarding header, so nothing
  coming through the tunnel reaches them. At most `MAX_CALLS` calls run at once.
- **Secrets.** Keys are read from `.env`, which is git-ignored; every log line is scrubbed of API keys and tokens.
- **Personal data.** Transcripts are logged only with `TRANSCRIPTS=1` and calls recorded only with `RECORD=1`, in which
  case the greeting says the call is recorded. Recordings, logs and `data/clinic.sqlite` stay on the machine and are
  git-ignored. Booking changes need the caller's name and phone number to match, and every change is audited.
- **Database.** All queries are parameterised.

## Before you deploy anything like this

The demo is not a production system. A real deployment needs, at least: a fixed public address instead of a
development tunnel, authentication on any staff interface, data processing agreements with the telephony, speech and
model providers, a retention policy for recordings and transcripts, and consent from everyone recorded.

## Known advisories

Pipecat 0.0.108 has an advisory for its development runner, which this project does not use.
