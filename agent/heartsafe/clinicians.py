"""Assumed clinician response model, used identically for HeartSafe and the Medly-alone baseline.

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
    resp = lambda: float(rng.exponential(c["response_after_reminder_h"]))  # noqa: E731
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
