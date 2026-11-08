"""LLM tool-calling orchestrator for case reviews.

Claude gets read-only tools over one case (the facts, data quality, ECG, MRI,
follow-up status) and writes the review by calling `submit_review`. It explains;
it never decides. Severity, the suggested action and deadlines come from the
deterministic policy in config.yaml and are copied into the output unchanged.
Every run is written to the case's audit log. Without credentials, or on any
error, the template review is returned instead.
"""
import json
import os

from .review import render

SYSTEM = (
    "You prepare case reviews for heart-failure remote-monitoring nurses. "
    "Call the tools to gather the facts for this case, then call submit_review exactly once. "
    "Use only facts returned by the tools: no new diagnoses, medications, numbers or recommendations. "
    "State every reliability caveat, and say plainly when ECG or MRI evidence was not used and why. "
    "The clinical team decides; never phrase anything as a decision or an order."
)

TOOLS = [
    {"name": "get_case_facts", "description": "Patient context, why Medly flagged the day, symptoms and the recent "
     "weight / blood pressure / heart rate trend.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_data_quality", "description": "Reading-level reliability flags (unit errors, cuff artifacts, "
     "scale changes, missed days) and notes from the data-quality checks.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_ecg", "description": "Single-lead ECG model result: abnormal probability, signal quality, "
     "uncertainty, whether it was reliable enough to use as evidence, and why not.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_mri", "description": "Cardiac MRI result: LVEF, scanner, segmentation uncertainty, "
     "plausibility and scanner-shift flags, caveats.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_followup", "description": "Case state, acknowledgement and resolution deadlines, reminders "
     "sent and the audit trail so far.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "submit_review", "description": "Submit the finished review. Call once, last.", "strict": True,
     "input_schema": {
         "type": "object", "additionalProperties": False,
         "required": ["summary", "why_flagged", "reliability_caveats", "evidence_used", "next_step_wording"],
         "properties": {
             "summary": {"type": "string", "description": "Two or three sentences for the nurse."},
             "why_flagged": {"type": "string"},
             "reliability_caveats": {"type": "array", "items": {"type": "string"}},
             "evidence_used": {"type": "string", "description": "Which inputs were trusted and which were not."},
             "next_step_wording": {"type": "string",
                                   "description": "The policy's suggested next step, reworded, same meaning."},
         }}},
]


def _tool_results(case, tracker):
    review = json.loads(case["review"]) if isinstance(case["review"], str) else case["review"]
    audit = tracker.audit(case["case_id"]) if tracker else []
    return review, {
        "get_case_facts": {k: review.get(k) for k in ("kind", "patient_id", "day", "context", "why_flagged",
                                                      "symptoms", "trend")},
        "check_data_quality": {"reliability_flags": review.get("reliability_flags"), "caveats": review.get("caveats")},
        "check_ecg": review.get("ecg") or {"note": "no ECG for this case"},
        "check_mri": review.get("mri") or {"note": "no MRI for this case"},
        "check_followup": {"state": case.get("state"), "severity": case.get("severity"),
                           "ack_deadline_h": case.get("ack_deadline_h"),
                           "resolve_deadline_h": case.get("resolve_deadline_h"),
                           "reminders": case.get("reminders"), "assigned_to": case.get("assigned_to"),
                           "audit": [{k: a[k] for k in ("t_h", "actor", "action")} for a in audit][-12:]},
    }


def _format(out, review):
    lines = [f"**{review['severity'].upper()} - {review['kind']} review for {review['patient_id']} "
             f"(day {review['day']})**", out["summary"], "Why flagged: " + out["why_flagged"],
             "Evidence: " + out["evidence_used"]]
    if out["reliability_caveats"]:
        lines.append("Reliability caveats: " + "; ".join(out["reliability_caveats"]))
    lines.append("Suggested next step: " + out["next_step_wording"])
    lines.append("HeartSafe does not make the clinical decision; the clinical team does.")
    return "\n\n".join(lines)


def orchestrate(case, tracker, model, client=None, now_h=None, max_turns=8):
    """Returns {"text", "source", "tool_calls"}; logs the run to the audit trail."""
    review, results = _tool_results(case, tracker)
    fallback = {"text": render(review), "source": "template", "tool_calls": []}

    def log(detail):
        if tracker is not None:
            tracker.log_llm(case["case_id"], now_h if now_h is not None else case.get("created_h", 0), detail)
            tracker.commit()

    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return fallback
        try:
            import anthropic
            client = anthropic.Anthropic()
        except Exception as e:  # package missing or misconfigured
            return {**fallback, "text": fallback["text"] + f"\n\n_(LLM unavailable: {type(e).__name__})_"}

    messages = [{"role": "user", "content": f"Prepare the review for case {case['case_id']}."}]
    calls = []
    try:
        for _ in range(max_turns):
            resp = client.beta.messages.create(
                model=model, max_tokens=4000, system=SYSTEM, tools=TOOLS, messages=messages,
                output_config={"effort": "medium"},
                # On a safety decline the API re-runs the request on a fallback model.
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            if resp.stop_reason == "refusal":
                raise RuntimeError("model declined")
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                raise RuntimeError(f"stopped ({resp.stop_reason}) without submit_review")
            messages.append({"role": "assistant", "content": resp.content})
            out, tool_results = None, []
            for u in uses:
                calls.append(u.name)
                if u.name == "submit_review":
                    out = u.input if isinstance(u.input, dict) else json.loads(u.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": u.id, "content": "received"})
                elif u.name in results:
                    tool_results.append({"type": "tool_result", "tool_use_id": u.id,
                                         "content": json.dumps(results[u.name], default=str)})
                else:
                    tool_results.append({"type": "tool_result", "tool_use_id": u.id, "is_error": True,
                                         "content": f"unknown tool {u.name}"})
            if out is not None:
                text = _format(out, review)
                log({"source": model, "tool_calls": calls})
                return {"text": text, "source": model, "tool_calls": calls}
            messages.append({"role": "user", "content": tool_results})
        raise RuntimeError("no review after the turn limit")
    except Exception as e:  # network, credentials, refusal, malformed output -> template, same flow
        log({"source": "template", "tool_calls": calls, "error": type(e).__name__})
        return {**fallback, "tool_calls": calls,
                "text": fallback["text"] + f"\n\n_(LLM review unavailable: {type(e).__name__}; template shown)_"}
