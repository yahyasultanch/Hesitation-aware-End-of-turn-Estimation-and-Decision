# HEED: Hesitation-aware End-of-turn Estimation and Decision

Turn-taking for voice agents that **waits only when it matters**. At every moment of silence HEED decides whether the
person has finished or is still mid-turn, and turns that into one decision, **respond, wait, ignore or yield**, always
with a reason code. It is built first for the people today's agents fail most, people who stutter, and it must not
slow everyone else down.

On stuttered speech HEED cuts in about a third as often as Pipecat's default turn detection, and it reaches the
cut-off rate of a long fixed timeout about half a second sooner on average. The package is `turnheed`; a Pipecat plugin
and a working phone agent are included.

## How it decides

1. **Sound** (`turnheed/wait.py`). Smart Turn v3.2's P(finished) and a still-talking model's P(mid-turn), plus how long
   real mid-turn pauses of that kind last, give P(turn ended | silence so far):

   P(ended | evidence, s) = π / (π + (1 − π) · S(s))

   where π comes from the two audio signals and S(s) is the share of real mid-turn pauses longer than s. The agent
   answers at the first moment this passes one threshold, learned from data: quick when the sound says finished
   (~0.3 s), up to ~3 s when it hears a block.
2. **This caller.** Each mid-turn pause the caller resumes from is remembered for this call only. A caller whose
   pauses run long earns longer waits; nobody's wait drops below the population's, so fluent callers are unaffected.
3. **Words.** Before replying, the agent's language model marks the caller's words complete or mid-sentence, and a
   reply to a mid-sentence turn is never spoken. The instructions discard repeated words and sounds, judge only the
   last words, and treat short replies and answers that trail off ("...so", "...yeah") as complete. If the caller
   stays quiet it checks again after 2.5 s (8 s for a caller who asked for time) and answers if it can.
4. **Fast path.** While a caller has had fewer than two mid-turn pauses over 1 s, the turn goes to the words check
   after 0.5 s unless the sound is sure they are mid-turn. Never while the caller reads out digits or spells.
5. **Noise never takes the turn.** Sound alone (line hiss, breath, a cough, a door) never starts a turn or stops the
   agent; a transcript does. While the agent speaks, a transcript of only backchannels ("mm", "okay", "yeah") is
   ignored, and any other word yields at once.

## Results

A **cut-off** is a reply inside a pause where the person was still mid-turn. A reply is **late** if it comes more than
2 s after a real turn end or the person carries on first. Mean reply is measured at turn ends that get a reply.
Intervals are 95%, resampling whole speakers or calls. All model signals are scored on speakers the models never saw.

**People who stutter** (FluencyBank, 36 adults; 2,000 mid-turn pauses, 210 turn ends):

| System | Cut-offs | Late | Mean reply |
|---|---|---|---|
| Smart Turn v3.2, Pipecat default | 20.9% [17.4-24.7] | 44.3% | 1.37 s |
| Fixed timeout 1.2 s | 16.3% [12.8-19.4] | 0.5% | 1.20 s |
| Fixed timeout 1.8 s | 6.4% | 3.3% | 1.80 s |
| HEED, sound only | 10.3% [7.0-13.5] | 7.1% | 1.04 s |
| **HEED, sound + words** | **7.8% [5.4-10.2]** | 10.5% | **1.12 s** |

**Real phone bookings** (SpokenWOZ test split, 1,000 calls at 8 kHz; 2,382 mid-turn pauses, 2,382 turn ends):

| System | Cut-offs | Late | Mean reply |
|---|---|---|---|
| Smart Turn v3.2, Pipecat default | 53.6% [51.2-55.9] | 23.4% | 0.65 s |
| Fixed timeout 1.2 s | 43.7% [41.2-46.1] | 2.0% | 1.20 s |
| Fixed timeout 1.8 s | 31.5% | 7.3% | 1.80 s |
| HEED, sound only | 43.7% [41.3-46.2] | 5.6% | 0.99 s |
| **HEED, sound + words** | **30.1% [28.0-32.4]** | 22.4% | **1.28 s** |

The words check here is Claude Haiku 5.5 with the instructions in `demos/phone/bot.py` (phone calls: an earlier
version of them). What the numbers show:

- **Waiting only when it matters.** A long fixed timeout can match HEED's cut-offs, but only by making every reply
  wait. At the same cut-off rate HEED answers about 0.55 s sooner on stuttered speech and 0.6 s sooner on phone calls
  (`bench/recipes/tradeoff_curves.py`).
- **The words carry it on the phone.** The sound of a pause barely separates thinking from finishing on booking calls
  (Smart Turn AUC 0.69); the words check does.
- **An open model works after calibration.** Qwen3 8B (Apache-2.0, run locally) with the same instructions, its
  P(complete) cut-point tuned on half the recordings and scored on the other half: stuttered 5.5% cut-offs [2.6-7.9]
  with 30.3% late; phone 32.7% [29.6-35.8] with 24.7% late (Haiku on the same halves: 4.3% / 53.9% and 29.8% /
  22.7%, earlier instructions). The cut-point must be tuned per domain.
- **Background noise** (`bench/recipes/noise_bench.py`, ESC-50 noise mixed into every gap): at 5 dB, stuttered cut-offs
  go 20.5% -> 30.0% for Pipecat's default and 5.0% -> 7.1% for HEED with words. Silero VAD heard speech in at most
  0.3% of silent pauses carrying the noise alone. In the transcripts (`bench/recipes/noise_stt.py`), noise changes
  30% of stuttered words, but the words check's verdicts shift in both directions and hold slightly more pauses.
- **Label audit** (`bench/recipes/label_audit.py`): of 200 hand-checked turn ends, the words check's "mid-sentence"
  calls on real ends were mostly its own errors, not label noise, which led to the current instructions.

## Install and use

```
pip install -e .            # the floor policy and learned wait: pure Python
pip install -e ".[pipecat]" # the Pipecat plugin
```

```python
from turnheed import FloorPolicy, Signals
from turnheed.wait import LearnedWait

pol = FloorPolicy(wait_model=LearnedWait.load("data/still_talking/wait_model_phone.json"), fast_wait=0.5)
d = pol.step(Signals(t=now, user_speaking=vad, p_finished=smart_turn_p, p_still_talking=hold_p))
d.decision, d.reason, d.hold_until     # e.g. "wait", "mid_turn_block", 12.4
```

Any `Signals` field left `None` drops out, so the same policy runs with or without each model.

```python
# Pipecat: Smart Turn + still-talking v2 + floor policy; noise cannot start a turn
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from turnheed.plugins.pipecat import FloorTurnAnalyzer, FloorTurnStartStrategy, FloorTurnStopStrategy
from turnheed.still_talking import StillTalkingV2

analyzer = FloorTurnAnalyzer(still_talking=StillTalkingV2(...))
strategies = UserTurnStrategies(start=[FloorTurnStartStrategy()], stop=[FloorTurnStopStrategy(turn_analyzer=analyzer)])

# LiveKit: scales any LiveKit turn detector's probability by a still-talking reading (no learned wait yet)
from turnheed.plugins.livekit import FloorTurnDetector
```

The model files (still-talking head, learned wait) are built by the recipes below and kept under `data/`; they are
not distributed, because the training data is research-only or non-commercial.

## Phone demo

`demos/phone/`: a physiotherapy booking line on Twilio that books, moves and cancels appointments in a SQLite database
through tool calls, in two modes with everything else identical: `baseline` (Pipecat's default) and `layer` (HEED).
See [demos/phone/README.md](demos/phone/README.md).

## Reproduce

Data goes under `data/` and never into git. Get FluencyBank (TalkBank account), Buckeye (registration, research use
only), SEP-28k (Apple's download scripts), SpokenWOZ (Hugging Face, `ssz1111/SpokenWOZ-Test-*`) and, for the noise
benchmark, ESC-50. Then:

```
pip install -e ".[bench,dev]"
python bench/recipes/fluencybank.py data/fluencybank          # bench files: speech runs, pauses, turn ends
python bench/recipes/buckeye.py data/buckeye
python bench/recipes/spokenwoz.py data/spokenwoz
python bench/recipes/smart_turn_signals.py data/<set>          # Smart Turn on every gap
python bench/recipes/smart_turn_embed.py data/fluencybank data/buckeye [--phone]
python bench/recipes/smart_turn_embed.py data/spokenwoz
python bench/recipes/still_talking_v2_train.py data            # still-talking v2, cross-validated by speaker
python bench/recipes/still_talking_v2_score.py data/spokenwoz
python bench/recipes/fit_wait.py data [--phone]                # learned wait
python bench/recipes/completion_veto_eval.py data/fluencybank 2000   # words check, Haiku (~$0.50)
JUDGE=ollama:qwen3:8b TAG=v2 python bench/recipes/completion_veto_eval.py data/spokenwoz   # local open model
BOOT=1000 python bench/recipes/completion_veto_eval.py data/fluencybank 2000              # with 95% intervals
python bench/recipes/tradeoff_curves.py                        # cut-offs vs reply time for every system
python bench/recipes/noise_bench.py data/fluencybank 2000      # ESC-50 noise in the audio models
python -m pytest -q
```

## Data and licences

| Data | Used for | Licence |
|---|---|---|
| FluencyBank (Voices-AWS, Timestamped) | stuttered benchmark, training | TalkBank ground rules |
| Buckeye Corpus | fluent benchmark, training | research use only |
| SEP-28k | still-talking v1 | CC BY-NC 4.0 |
| SpokenWOZ (test split) | held-out phone-call benchmark | CC BY-NC 4.0 |
| ESC-50 | noise benchmark | CC BY-NC 3.0 |
| Smart Turn v3.2 (via Pipecat) | P(finished), encoder | BSD-2 |

The code is Apache-2.0. Trained weights are not distributed, and models trained on these datasets are for research
and non-commercial use.

## Limitations

- No person who stutters has evaluated the agent yet: everything above comes from recorded speech.
- The words check reads corpus transcripts in the benchmark, not live recognition output (`noise_stt.py` is the
  exception). The stuttered and fluent sets differ in recording setup.
- On fluent conversational interviews (Buckeye) the sound-only layer is no better than a timeout, and Qwen3 8B fails
  as a words check on that unpunctuated text; a stronger model is needed there.
- The 100 ms median-added-delay target is not met: below 0.5 s, cut-offs rise.
- The LiveKit plugin lacks the learned wait and the words check.

## Security

See [SECURITY.md](SECURITY.md). Never commit `.env`, recordings, transcripts or the clinic database; all are
git-ignored.

## Licence

Apache-2.0, see [LICENSE](LICENSE).
