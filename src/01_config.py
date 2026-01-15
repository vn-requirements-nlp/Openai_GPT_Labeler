# src/config.py
from __future__ import annotations

# 12 nhãn bạn yêu cầu (PROMISE-style)
LABELS = [
    ("F",  "Functional (F)",         "What the system should do (functional behavior)."),
    ("A",  "Availability (A)",       "Uptime, availability, reliability in terms of being reachable."),
    ("FT", "Fault Tolerance (FT)",   "Graceful degradation, recovery, resilience to faults."),
    ("L",  "Legal (L)",              "Compliance, licensing, legal constraints."),
    ("LF", "Look & Feel (LF)",       "UI aesthetics, layout, visual appearance."),
    ("MN", "Maintainability (MN)",   "Ease of maintenance, modification, testability."),
    ("O",  "Operability (O)",        "Ease of operating/administering the system, monitoring, manageability."),
    ("PE", "Performance (PE)",       "Speed, latency, throughput, resource usage."),
    ("PO", "Portability (PO)",       "Compatibility across platforms/environments."),
    ("SC", "Scalability (SC)",       "Ability to scale with load/users/data."),
    ("SE", "Security (SE)",          "Authn/authz, encryption, confidentiality, integrity."),
    ("US", "Usability (US)",         "Ease of use, UX, learnability, accessibility."),
]

CODE_TO_COL = {code: col for code, col, _ in LABELS}
COL_TO_CODE = {col: code for code, col, _ in LABELS}
LABEL_COLS = [col for _, col, _ in LABELS]

LABELS_DESC = "\n".join([f"- {code}: {col} — {desc}" for code, col, desc in LABELS])

SYSTEM_PROMPT = f"""
You are an expert in software requirements classification (PROMISE NFR style).
Classify the given requirement into the 12 labels below.

LABELS (multi-label allowed; set multiple labels to 1 if clearly applicable):
{LABELS_DESC}

Rules:
- Output MUST be multi-label: each label is 0 or 1.
- Set a label to 1 only if the requirement explicitly or strongly implies it.
- If unsure, prefer 0 (be conservative).
- Functional (F) can co-exist with NFR labels if the requirement includes both what + quality constraints.
"""
