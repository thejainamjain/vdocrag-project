"""
PDF page-count policy for the "Upload & Index" tab.

Not a VRAM constraint — indexing processes one page at a time, so VRAM
doesn't scale with page count. This is a wall-clock UX / Colab-idle-timeout
concern: how long is it reasonable to make someone wait, with or without a
progress bar, before the demo either warns them or asks for confirmation.

`observed_s_per_page` should come from real telemetry (see
`observed_seconds_per_page_from_log` below) once Step 1 / any real indexing
run has produced a log file. Until then it's seeded with an estimate derived
from the paper's own A100 numbers (Table 7: 204.4ms/page) scaled by a rough
4-6x T4-vs-A100 slowdown factor — see handoff doc Section 5.4 for the full
reasoning. Treat the seed as order-of-magnitude, not measured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# What "no warning needed" and "hard-block by default" mean in wall-clock time.
TARGET_MAX_WAIT_S = 120
WARN_MAX_WAIT_S = 480

# Provisional estimate pending real Step 1 / indexing-run data.
SEED_S_PER_PAGE = 1.2


def page_cap(observed_s_per_page: float = SEED_S_PER_PAGE) -> dict:
    """Returns the two thresholds (in page count) that drive the UI:
      - <= no_warning_pages: index immediately, just show a progress bar.
      - between the two: confirm-before-start banner with a time estimate.
      - > warn_pages: blocked by default, with an explicit "index anyway"
        opt-in rather than a hard wall (testing large-corpus / multi-hop
        behavior is a legitimate use case for this tool).
    """
    if observed_s_per_page <= 0:
        raise ValueError("observed_s_per_page must be positive")
    return {
        "observed_s_per_page": observed_s_per_page,
        "no_warning_pages": max(1, int(TARGET_MAX_WAIT_S / observed_s_per_page)),
        "warn_pages": max(1, int(WARN_MAX_WAIT_S / observed_s_per_page)),
    }


def estimated_wait_s(num_pages: int, observed_s_per_page: float = SEED_S_PER_PAGE) -> float:
    return round(num_pages * observed_s_per_page, 1)


def observed_seconds_per_page_from_log(log_path: str | Path) -> Optional[float]:
    """Recompute the real per-page encoding time from a telemetry JSONL log
    (see telemetry.py), using the median over successful `encode_document`
    calls. Returns None if the log has no matching entries yet (e.g. before
    any real indexing run) so callers can fall back to SEED_S_PER_PAGE.
    """
    import pandas as pd

    log_path = Path(log_path)
    if not log_path.exists():
        return None

    df = pd.read_json(log_path, lines=True)
    if df.empty or "component" not in df.columns:
        return None

    enc = df[
        (df["component"] == "retriever")
        & (df["function"] == "encode_document")
        & (df["status"] == "ok")
    ]
    if enc.empty:
        return None

    return round(enc["duration_ms"].median() / 1000, 3)
