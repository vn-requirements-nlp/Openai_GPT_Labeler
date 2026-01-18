from __future__ import annotations

# Định nghĩa nâng cao với Keywords và Decision Boundaries
LABELS_DETAILED = [
    {
        "code": "F", "name": "Functional",
        "desc": "Functionality, typical 'The system shall do X'. Input/Output behaviors.",
        "keywords": ["process", "calculate", "store", "display", "send"],
        "note": "Can coexist with NFRs (e.g., 'Authenticate user' is F and SE)."
    },
    {
        "code": "A", "name": "Availability",
        "desc": "System uptime, availability windows, status monitoring.",
        "keywords": ["uptime", "24/7", "99.9%", "available", "downtime"],
        "note": "Distinct from FT. A is about 'being up'. FT is about 'handling crashes'."
    },
    {
        "code": "FT", "name": "Fault Tolerance",
        "desc": "Error handling, recovery, resilience against failure.",
        "keywords": ["recover", "crash", "backup", "restore", "failover", "exception"],
        "note": "Focus on the REACTION to failure, not just the uptime stat."
    },
    {
        "code": "L", "name": "Legal",
        "desc": "Legislative, copyright, licensing, compliance requirements.",
        "keywords": ["law", "GDPR", "license", "copyright", "terms", "policy"],
        "note": "Strict legal constraints, not just general security rules."
    },
    {
        "code": "LF", "name": "Look & Feel",
        "desc": "Aesthetics, UI design styles, colors, branding.",
        "keywords": ["color", "font", "logo", "style", "theme", "layout"],
        "note": "Distinct from US. LF is about 'beauty/branding'. US is about 'easy of use'."
    },
    {
        "code": "MN", "name": "Maintainability",
        "desc": "Code quality, modularity, extensibility, installation easy for devs.",
        "keywords": ["modular", "code", "update", "patch", "extensible", "document"],
        "note": "Focus on the DEVELOPER experience, not the end-user."
    },
    {
        "code": "O", "name": "Operability",
        "desc": "Admin tasks, system monitoring, configuration, deployment management.",
        "keywords": ["admin", "monitor", "log", "config", "deploy", "manage"],
        "note": "Focus on the SYSTEM ADMIN / DEVOPS experience."
    },
    {
        "code": "PE", "name": "Performance",
        "desc": "Response time, throughput, resource utilization (RAM/CPU).",
        "keywords": ["second", "ms", "response time", "throughput", "latency", "concurrent"],
        "note": "Performance under NORMAL load. Scaling (SC) is about INCREASING load."
    },
    {
        "code": "PO", "name": "Portability",
        "desc": "Running on different OS, browsers, devices, locations.",
        "keywords": ["Windows", "Linux", "browser", "mobile", "compatible", "device"],
        "note": "Adaptability to different environments."
    },
    {
        "code": "SC", "name": "Scalability",
        "desc": "Handling growth in users, data, or traffic volume.",
        "keywords": ["scale", "growth", "expand", "increase users", "volume"],
        "note": "Ability to handle MORE over time without redesign."
    },
    {
        "code": "SE", "name": "Security",
        "desc": "Authentication, authorization, encryption, data privacy.",
        "keywords": ["login", "password", "encrypt", "auth", "permission", "access"],
        "note": "Protecting data and access."
    },
    {
        "code": "US", "name": "Usability",
        "desc": "Easy of learning, user efficiency, error prevention for users.",
        "keywords": ["easy", "intuitive", "click", "user friendly", "learn", "help"],
        "note": "Distinct from LF. US is about interaction efficiency."
    },
]

# Re-build simple structures for compatibility
LABELS = [(x["code"], x["name"], x["desc"]) for x in LABELS_DETAILED]
CODE_TO_COL = {x["code"]: f"{x['name']} ({x['code']})" for x in LABELS_DETAILED}
COL_TO_CODE = {v: k for k, v in CODE_TO_COL.items()}
LABEL_COLS = list(CODE_TO_COL.values())

# Generate optimized description block
def build_label_desc():
    lines = []
    for item in LABELS_DETAILED:
        lines.append(f"- **{item['name']} ({item['code']})**: {item['desc']}")
        lines.append(f"  *Keywords*: {', '.join(item['keywords'])}")
        lines.append(f"  *Distinction*: {item['note']}")
    return "\n".join(lines)

SYSTEM_PROMPT = f"""
You are an expert Requirements Engineer specializing in NFR classification using the ISO/IEC 25010 and PROMISE framework.

### TASK
Analyze the software requirement provided by the user and classify it into the 12 categories below.
Focus on the *intent*, *context*, and *explicit constraints* imposed by the requirement.

### CATEGORY DEFINITIONS
{build_label_desc()}

### ANALYSIS STRATEGY (Chain-of-Thought)
1. **Identify Keywords**: Scan for keywords defined in the definitions.
2. **Determine Intent**:
   - Is it describing *what* the system does? -> Functional (F).
   - Is it describing *how well* (speed, uptime, security)? -> NFRs.
3. **Resolve Conflicts**:
   - **Availability (A) vs Fault Tolerance (FT)**: 'A' is about uptime statistics. 'FT' is about recovery mechanisms (backups, redundancies).
   - **Usability (US) vs Look & Feel (LF)**: 'US' is about interaction flow/efficiency. 'LF' is about aesthetics/colors.
   - **Performance (PE) vs Scalability (SC)**: 'PE' is speed under current load. 'SC' is ability to handle future growth.
4. **Final Decision**: 
   - A requirement can be multi-label.
   - Be conservative: explicit evidence is required for 'True'.

### OUTPUT INSTRUCTION
Provide your reasoning first in the 'rationale' field, then set the boolean flags.
"""