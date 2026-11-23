from turnheed import IGNORE, RESPOND, WAIT, YIELD, FloorPolicy, Signals
from turnheed.bench import DEFAULT_SYSTEMS, run, score, timeout_baseline, policy_system


def silence_run(pol, start, end, **kw):
    """Step through silence from start to end; return first non-wait decision."""
    t = start
    while t < end:
        d = pol.step(Signals(t=round(t, 3), user_speaking=False, **kw))
        if d.decision != WAIT:
            return t, d
        t += 0.01
    return None, d


def test_audio_only_timeout_and_block_hold():
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 2.0)
    assert d.decision == RESPOND and 0.5 <= t <= 0.52

    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 1.7, p_still_talking=0.9, still_talking_kind="block")
    assert t is None and d.reason == "mid_turn_block"
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 3.0, p_still_talking=0.9, still_talking_kind="block")
    assert abs(t - (0.5 + 0.6 * 2.197)) < 0.02  # wait = timeout + tau * logit(p_still)


def test_hold_ceiling_and_context():
    pol = FloorPolicy(max_hold=1.0)
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 5.0, p_still_talking=0.9)
    assert d.reason == "hold_ceiling" and abs(t - 1.0) < 0.02
    pol = FloorPolicy(max_hold=1.0)
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 5.0, p_still_talking=0.9, expected="digits")
    assert abs(t - 1.6) < 0.02


def test_turn_detector_and_still_talking_fusion():
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 2.0, p_finished=0.9)
    assert d.decision == RESPOND and t < 0.25
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 2.0, p_finished=0.9, mouth_moving=0.9)  # camera sees mouth working
    assert abs(t - 0.5) < 0.02  # conflicting evidence cancels: back to the plain timeout


def test_yield_backchannel_side_talk():
    pol = FloorPolicy()
    assert pol.step(Signals(t=0, user_speaking=True, agent_speaking=True, p_barge_in=0.9)).decision == YIELD
    assert pol.step(Signals(t=0, user_speaking=True, agent_speaking=True, p_barge_in=0.1)).reason == "backchannel"
    pol = FloorPolicy()
    assert pol.step(Signals(t=0, user_speaking=True, p_addressed=0.1)).reason == "side_talk"
    _, d = silence_run(pol, 0.01, 2.0, p_finished=0.9)
    assert d.decision == IGNORE  # never answer a turn that wasn't for us
    pol = FloorPolicy()
    assert pol.step(Signals(t=0, user_speaking=True, face_speaking=False)).reason == "off_camera_voice"


def test_cutoff_recovery_raises_session_patience():
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 2.0)
    d = pol.step(Signals(t=t + 0.3, user_speaking=True))
    assert d.decision == YIELD and d.reason == "cutoff_recovery" and pol.patience > 1.0


def test_bench_metrics():
    rec = {"id": "x", "condition": "stuttered", "segments": [
        {"start": 0.0, "end": 1.0, "pause_type": "block"},
        {"start": 2.0, "end": 3.0, "end_of_turn": True}]}
    r = score(rec, timeout_baseline(500))
    assert r["cutoffs"] == 1 and r["split_turns"] == 1 and abs(r["delays"][0] - 0.5) < 0.02
    r = score(rec, policy_system(True))
    assert r["cutoffs"] == 0 and r["late"] == 0
    rows = run([rec], DEFAULT_SYSTEMS)
    assert len(rows) == len(DEFAULT_SYSTEMS)


def test_patience_relaxes_after_good_reply():
    pol = FloorPolicy()
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 2.0)
    pol.step(Signals(t=t + 0.3, user_speaking=True))          # cut-off: patience up
    raised = pol.patience
    t, _ = silence_run(pol, t + 0.31, t + 5.0)                # they finish, we reply
    silence_run(pol, t + 0.01, t + 2.0)                       # nobody resumes: reply stood
    assert 1.0 < pol.patience < raised


def test_bench_signal_files():
    from turnheed.bench import MODEL_SYSTEMS
    rec = {"id": "x", "condition": "stuttered", "segments": [
        {"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0, "end_of_turn": True}]}
    hold = {"x": {"0": {"p_still_talking": 0.9, "still_talking_kind": "block"}, "1": {"p_still_talking": 0.1}}}
    r = score(rec, MODEL_SYSTEMS["policy + still_talking"], hold)
    assert r["cutoffs"] == 0 and r["late"] == 0
    st = {"x": {"0": {"p_finished": 0.9}, "1": {"p_finished": 0.9}}}
    r = score(rec, MODEL_SYSTEMS["smart_turn (pipecat default)"], st)
    assert r["cutoffs"] == 1 and abs(r["delays"][0] - 0.2) < 0.02  # answers when VAD stop hands it the audio


def test_learned_wait_is_targeted_and_only_extends_per_call():
    from turnheed.wait import LearnedWait
    grid = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    S = [[1.0, 1.0, 0.2, 0.1, 0.05, 0.02, 0.01]] * 2             # mid-turn pauses rarely pass 1 s
    w = LearnedWait([1.0, -1.0, 0.0], [0.0], grid, S, theta=0.5)
    fast, held = w(0.9, 0.1), w(0.3, 0.7)
    assert fast <= 0.5 < held < 5.0                             # finished -> fast, mid-turn -> wait
    assert w(0.3, 0.7, (0.2, 0.3, 0.2, 0.25)) >= held           # short pauses never shorten the wait
    assert w(0.3, 0.7, (2.5, 2.8, 3.5, 2.6, 2.9)) > held        # a caller with long blocks earns longer waits
    pol = FloorPolicy(wait_model=w)
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 5.0, p_finished=0.9, p_still_talking=0.1)
    assert d.decision == RESPOND and t <= 0.55


def test_fast_path_is_targeted():
    pol = FloorPolicy(fast_wait=0.5)                       # fresh caller: hand over early
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.5)
    assert d.decision == RESPOND and t <= 0.52
    pol = FloorPolicy(fast_wait=0.5)                       # audio sure they're mid-turn: no fast path
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.95)
    assert t is None or t > 0.6
    pol = FloorPolicy(fast_wait=0.5)                       # a caller who has had two long pauses: off
    pol.pauses = [1.4, 2.2]
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.5)
    assert t is None or t > 0.6


def test_fast_path_never_for_numbers_or_after_resumes():
    pol = FloorPolicy(fast_wait=0.5)                       # agent asked for a number: no fast path
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.5, expected="digits")
    assert t is None or t > 0.6
    pol = FloorPolicy(fast_wait=0.5)                       # caller twice carried on right after a hand-over: off
    pol.events["cutoff_recovery"] = 2
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.5)
    assert t is None or t > 0.6


def test_question_hands_over_even_in_patient_mode():
    # live call: patient caller (long pauses, asked for time) said "are you there?" and waited 7 s
    pol = FloorPolicy(fast_wait=0.5, max_hold=5.0)
    pol.pauses, pol.patience = [1.4, 2.2], 1.5
    pol.step(Signals(t=0.0, user_speaking=True))
    t, d = silence_run(pol, 0.01, 8.0, p_finished=0.3, p_still_talking=0.95, asked=True)
    assert d.decision == RESPOND and t <= 0.52
    pol = FloorPolicy(fast_wait=0.5)                       # but not while reading out a number
    pol.step(Signals(t=0.0, user_speaking=True))
    t, _ = silence_run(pol, 0.01, 3.0, p_finished=0.3, p_still_talking=0.5, expected="digits", asked=True)
    assert t is None or t > 0.6
