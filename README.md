# HeartSafe Agent — WAI Canada Hackathon 2026 (Team #40)

Equity-aware trust and coordination layer for remote cardiac monitoring (e.g. Medly, UHN): reliability checks,
explainable reviews, tracked follow-up and equity monitoring, so existing monitoring is safer and accountable.

| Folder | What | Owner |
| --- | --- | --- |
| `imaging/` | Cardiac MRI segmentation, imaging reliability checks, imaging equity audit | Yassmin |
| `ecg/` | Single-lead ECG classifier, ECG reliability checks | Ana Karen |
| `agent/` | Coordination layer: Medly adapter, reliability, reviews, case tracker, equity monitor, dashboard | Vertika |

Every model writes one JSON per exam with
`{prediction, probability, uncertainty, quality, shift_score, metadata, flags, suggested_action}`;
see `imaging/README.md` for the imaging fields. Data is never committed (see `.gitignore`).

Quick start: `cd agent && pip install -r requirements.txt && python -m heartsafe.run && streamlit run app.py`.
