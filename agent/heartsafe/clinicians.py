"""Assumed clinician response model, used identically for HeartSafe and the platform-alone baseline.

Without tracking, an alert can slip through (forget_prob) and is never acted on.
With HeartSafe, a missed deadline triggers a reminder, and a second miss goes to a
backup clinician. These are simulation assumptions, listed in config.yaml.
"""
import numpy as np


def _lognormal(rng, median, sigma):
    return float(rng.lognormal(np.log(median), sigma))


def followup_times(severity, remoteness, ack_window_h, resolve_window_h, cfg, rng, reminders, force_forget=False):
    """Hours after creation for acknowledge / resolve, and reminders sent.

    Returns dict(ack, resolve, reminders: list of (t, to), lost) with ack/resolve None if never done.
    """
    c = cfg["clinicians"]
    natural = _lognormal(rng, c["median_ack_h"][severity][remoteness], c["ack_sigma"])
    forgot = force_forget or rng.random() < c["forget_prob"]
    after = c.get("emergency_response_after_reminder_h", c["response_after_reminder_h"]) \
        if severity == "emergency" else c["response_after_reminder_h"]
    resp = lambda: float(rng.exponential(after))  # noqa: E731
    sent = []
    if not reminders:
        if forgot:
            return {"ack": None, "resolve": None, "reminders": sent, "lost": True, "forgot": True}
        ack = natural
    elif forgot or natural > ack_window_h:
        sent.append((ack_window_h, "primary"))
        if forgot and rng.random() < c["ignore_reminder_prob"]:
            sent.append((2 * ack_window_h, "backup"))
            ack = 2 * ack_window_h + resp()
        else:
            ack = ack_window_h + resp() if forgot else min(natural, ack_window_h + resp())
    else:
        ack = natural
    resolve = ack + _lognormal(rng, c["median_resolve_after_ack_h"][severity], 0.6)
    if reminders and resolve > resolve_window_h:
        sent.append((resolve_window_h, "primary"))
        resolve = max(ack + 0.5, min(resolve, resolve_window_h + 2 * resp()))
    # forgot = the case would have stalled without tracking (ground truth for the evaluation)
    return {"ack": ack, "resolve": resolve, "reminders": sent, "lost": False, "forgot": forgot}


def patient_response(severity, patient, confirm_window_h, cfg, rng, force_no_response=False):
    """How the patient side responds to a clinician's instruction (simulation assumptions in config.yaml).

    Returns hours after the instruction is sent: confirm (and who confirmed), action, the follow-up
    contacts HeartSafe makes, and `fail`/`fail_reason` when the loop cannot be completed.
    gt_self_completed: the patient reads the message on their own (even late) and acts without a
    reminder, i.e. what the platform alone (one message, no tracking) would achieve. Evaluation only.
    """
    L = cfg["patient_loop"]
    rem = patient["remoteness"]
    helper = "caregiver" if patient.get("caregiver") else "community_health_worker"
    out = {"contacts": [], "confirm": None, "by": None, "action": None, "fail": None, "fail_reason": None}
    own = None
    if not force_no_response and rng.random() >= L["patient_no_response"][rem]:
        own = _lognormal(rng, L["patient_median_confirm_h"][rem], 0.7)
    if own is not None and own <= confirm_window_h:
        out["confirm"], out["by"] = own, "patient"
    else:
        t = confirm_window_h
        helper_win = max(confirm_window_h, L["helper_window_h"])
        for who in ([helper, "community_health_worker"] if helper == "caregiver" else [helper, helper]):
            out["contacts"].append((t, who))
            reached = rng.random() < L["helper_reach"] or force_no_response  # demo: the helper is reached
            t_help = t + _lognormal(rng, L["helper_median_confirm_h"], 0.6) if reached else None
            cands = [(x, w) for x, w in ((own, "patient"), (t_help, who)) if x is not None]
            if cands and min(cands)[0] <= t + helper_win:
                out["confirm"], out["by"] = min(cands)
                break
            t += helper_win
        if out["confirm"] is None:
            out["fail"], out["fail_reason"] = t, "patient could not be reached"
            out["gt_self_completed"] = False
            return out
    natural = rng.random() >= L["action_not_done"]
    a_win = L["action_deadline_h"][severity]
    tc = out["confirm"]
    if natural:
        out["action"] = tc + _lognormal(rng, L["median_action_h"][severity], 0.6)
    if out["action"] is None or out["action"] > tc + a_win:
        late = out["action"]
        out["contacts"].append((tc + a_win, out["by"]))
        if rng.random() < L["action_after_reminder"]:
            out["action"] = min(late, tc + a_win + 6) if late else tc + a_win + float(rng.exponential(6))
        else:
            out["action"] = None
            out["fail"], out["fail_reason"] = tc + 2 * a_win, "instruction not completed"
    out["gt_self_completed"] = own is not None and natural
    return out
