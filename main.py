#!/usr/bin/env python3
"""Neon Coastal Rarity Forecast — GTA VI trophy earn-rate forecaster.

Reads every *.in case file in the input dir and writes one <id>.out per
case: five quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate
at the case target. Stdlib only, no network, fully deterministic.

Model (generalises, never hardcodes any refresh):
  rate = earners / players_tracked.  For each case we fit, on the reads of
  the current regime (after the last big jump in players_tracked, else the
  last 24 h, else all reads), a power law in hours since release,
  log(P) = a + b*log(1 + x) for players and log(E_j) = a_j + b_j*log(1 + x)
  for each trophy's earners (E_j = rate_j * P / 100).  We predict P and E_j
  at the target and take rate_j = E_j / P.  When a series is flat (no growth
  signal, e.g. launch-night practice) we use a launch-surge prior instead
  of pretending nothing will change: players keep growing (b = 1.5) while
  each trophy's earners grow at the players log-rate scaled by the trophy's
  current rate — a 100% trophy is earned by every newcomer so it stays at
  100%, a rare one by almost none so its rate falls.  Quantile spread lives
  s = sqrt((0.5*sqrt(days))^2 + fit_residual^2), q_tau =
  inv_logit(logit(median) + z_tau * s).

Consistency (applied to the forecast medians, whole quantile vectors
shifted in logit space so spreads are kept):
  - Platinum (id 1) is the rarest trophy by definition (it needs every
    other trophy): each of its quantiles is capped at the lowest other
    trophy's same-level quantile.
  - Story progression is monotone: Gone Fishing, Chapter 1..4, final
    mission (ids 2,3,4,5,6,7) never rise along the sequence.
  - Long-grind / collect-everything trophies never exceed the hardest
    story chapter's rate (cap = max median of ids 3..7).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Story chain in play order: first mission, chapters 1-4, final mission.
STORY_IDS = [2, 3, 4, 5, 6, 7]
# Long-grind / collect-everything trophies (by description): they sit below
# the story chapters, never above the hardest chapter's rate.
GRIND_IDS = {14, 15, 17, 18, 19, 25, 28, 30, 31, 32, 35, 42, 54, 55}

# Launch-surge prior used only when a series shows no growth in history.
PRIOR_PLAYERS_SLOPE = 1.5
MIN_FIT_RESID = 0.30
MAX_FIT_RESID = 1.00


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    q = clamp(p, 0.01, 99.99) / 100.0
    return math.log(q / (1.0 - q))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def linfit(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        var = 0.0
    return a, b, math.sqrt(var)


def regime_reads(reads):
    """Reads of the current regime: after the last big players jump.

    A jump is players_tracked rising >50% between consecutive reads
    (an Exophase refresh catching up).  We fit on the regime after the
    last jump when it has >= 3 reads, else the last 24 h, else all reads.
    """
    idx = 0
    for i in range(1, len(reads)):
        p0 = reads[i - 1][1]
        p1 = reads[i][1]
        if p0 and p0 > 0 and p1 > 1.5 * p0:
            idx = i
    if len(reads) - idx >= 3:
        return reads[idx:]
    if not reads:
        return reads
    last_t = reads[-1][0]
    recent = [r for r in reads if (last_t - r[0]).total_seconds() <= 24 * 3600]
    if len(recent) >= 3:
        return recent
    return reads


def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    target_t = parse_time(case["target"])
    x_target = math.log1p(max(hours_since_release(target_t), 0.0))

    # (time, players, rates) triples, oldest first.
    reads = []
    for h in case.get("history", []):
        try:
            t = parse_time(h["at"])
            p = h.get("players_tracked")
            rates = h.get("rates") or []
        except Exception:
            continue
        if p is None or p <= 0:
            continue
        reads.append((t, float(p), rates))
    reads.sort(key=lambda r: r[0])
    if not reads:
        # No history at all: fall back to a flat forecast at 0.5%.
        return [{"id": t["id"], "q": [0.5, 0.5, 0.5, 0.5, 0.5]}
                for t in trophies]

    reg = regime_reads(reads)
    xs = [math.log1p(max(hours_since_release(t), 0.0)) for t, _, _ in reg]

    # --- players model ---
    ps = [p for _, p, _ in reg]
    if len(set(ps)) >= 2 and len(xs) >= 2:
        a_p, b_p, _ = linfit(xs, [math.log(p) for p in ps])
    else:
        # Flat players: intercept anchored at the last read, then the
        # launch-surge prior slope from there.
        b_p = PRIOR_PLAYERS_SLOPE
        a_p = (math.log(ps[-1]) - b_p * xs[-1]) if ps and xs else 0.0
    log_p_target = a_p + b_p * x_target
    p_target = max(math.exp(log_p_target), ps[-1], 1.0)

    # --- per-trophy earners model ---
    s_rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    rows = {}  # id -> [q05,q25,q50,q75,q95] in percent
    for j, tr in enumerate(trophies):
        pts = []
        for k, (_, p, rates) in enumerate(reg):
            if j >= len(rates):
                continue
            r = rates[j]
            if r is None:
                continue
            e = float(r) / 100.0 * p
            if e > 0:
                pts.append((xs[k], math.log(e)))
        last_rate = None
        for _, _, rates in reversed(reg):
            if j < len(rates) and rates[j] is not None:
                last_rate = float(rates[j])
                break
        if last_rate is None:
            last_rate = 0.5
        if len(pts) >= 2 and len(set(y for _, y in pts)) >= 2:
            a_j, b_j, resid = linfit([x for x, _ in pts],
                                     [y for _, y in pts])
            s_fit = clamp(resid, MIN_FIT_RESID, MAX_FIT_RESID)
        else:
            # Flat earners: a newcomer earns this trophy with probability
            # ~ its current rate (a 100% trophy is earned by everyone, a
            # rare one by almost no newcomer), so the earners log-slope is
            # the players slope scaled by the last rate. Intercept anchored
            # at the last read.
            frac = clamp(last_rate, 0.01, 99.99) / 100.0
            b_j = b_p * frac
            if pts:
                a_j = pts[-1][1] - b_j * pts[-1][0]
            else:
                a_j = math.log(max(last_rate, 0.01) / 100.0 * ps[-1])
            s_fit = MIN_FIT_RESID
        log_e_target = a_j + b_j * x_target
        med = 100.0 * math.exp(log_e_target) / p_target
        med = clamp(med, 0.01, 99.99)
        s = math.sqrt(s_rw * s_rw + s_fit * s_fit)
        lm = logit_pct(med)
        qs = [round(inv_logit_pct(lm + z * s), 4) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows[tr["id"]] = [clamp(q, 0.01, 99.99) for q in qs]

    # --- consistency: shift whole vectors in logit space ---
    def shift_vector(tid, new_q50):
        q = rows[tid]
        old = q[2]
        if old <= 0 or new_q50 >= old:
            return
        d = logit_pct(old) - logit_pct(new_q50)
        rows[tid] = [clamp(inv_logit_pct(logit_pct(v) - d), 0.01, 99.99)
                     for v in q]

    # Platinum <= rarest other trophy, at every quantile level.
    if 1 in rows:
        others = [v for k, v in rows.items() if k != 1]
        if others:
            for k in range(5):
                cap = min(v[k] for v in others)
                if rows[1][k] > cap:
                    rows[1][k] = cap
            for k in range(1, 5):
                if rows[1][k] < rows[1][k - 1]:
                    rows[1][k] = rows[1][k - 1]

    # Story monotone: medians never rise along the chain.
    prev = None
    for tid in STORY_IDS:
        if tid not in rows:
            continue
        q50 = rows[tid][2]
        if prev is not None and q50 > prev:
            shift_vector(tid, prev)
            q50 = rows[tid][2]
        prev = q50

    # Grind / collectibles never exceed the hardest story chapter.
    story_caps = [rows[t][2] for t in STORY_IDS[1:] if t in rows]
    if story_caps:
        cap = max(story_caps)
        for tid in GRIND_IDS:
            if tid in rows and rows[tid][2] > cap:
                shift_vector(tid, cap)

    # Final cleanup: clamp + non-decreasing + 4 decimals.
    out = []
    for tr in trophies:
        q = rows.get(tr["id"])
        if q is None:
            q = [0.5, 0.5, 0.5, 0.5, 0.5]
        q = [round(clamp(v, 0.01, 99.99), 4) for v in q]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        out.append({"id": tr["id"], "q": q})
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        doc = {"trophies": rows}
        text = json.dumps(doc, separators=(",", ":"))
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
