# HeartSafe Agent — WAI Canada Hackathon 2026 (Team #40)

Equity-aware AI agent for trustworthy cardiac monitoring and closed-loop follow-up in rural Canada.

| Folder | What | Owner |
| --- | --- | --- |
| `imaging/` | Cardiac MRI segmentation, imaging reliability checks, imaging equity audit | Yassmin |
| `ecg/` | Single-lead ECG classifier, ECG reliability checks | Ana Karen |
| `agent/` | Orchestrator, case tracker, dashboard | Vertika |

Every model writes one JSON per exam with
`{prediction, probability, uncertainty, quality, shift_score, metadata, flags, suggested_action}`;
see `imaging/README.md` for the imaging fields. Data is never committed (see `.gitignore`).
