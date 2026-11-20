#!/usr/bin/env python3
"""Neon Coastal Rarity Forecast v2 — GTA VI trophy earn-rate forecaster.

Reads every *.in case file in the input dir and writes one <id>.out per
case: five quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate
at the case target. Stdlib only, no network, fully deterministic.

Model — regime-aware, and it never hardcodes any refresh (no practice
numbers anywhere: no player count, no growth factor, no rate is baked in).
  rate = earners / players_tracked.  The regime is detected from the
  case's own history:
  * GROWTH regime (players_tracked moves): fit, on the reads of the
    current regime (after the last big jump in players_tracked, else the
    last 24 h, else all reads), a power law in hours since release,
    log(P) = a + b*log(1 + x) for players and log(E_j) = a_j + b_j*log(1
    + x) for each trophy's earners (E_j = rate_j * P / 100).  Predict P
    and E_j at the target and take rate_j = E_j / P.  This is the regime
    the final case (f00) lives in: its history already holds the
    post-refresh reads, so the growth is visible and fitted.
  * STALE regime (players_tracked flat across every read — the page is a
    frozen Exophase snapshot, as on launch night): the displayed count is
    as old as the FIRST read, so the players multiplier is modelled with
    a launch-night exponential prior — tracked players roughly double
    every 2 hours in the opening night — applied over the STALENESS
    window (first read -> target), capped at 12x.  Earners grow only via
    newcomers: E_target = E_last * (1 + (G-1)*frac), frac = the trophy's
    current rate, so a 100% trophy is earned by every newcomer and stays
    at 100%, while a rare one's earners barely move and its rate falls by
    ~the players multiplier.  The stale window is a lower bound (the page
    may already have been stale at the first read), so the dilution is a
    conservative estimate, and the spread below covers the rest.
  Quantile spread lives on the logit scale: s = sqrt((0.5*sqrt(days))^2
  + s_fit^2).  In the stale regime s_fit = 0.50, an honest width for a
  refresh whose size is uncertain within about a factor of two either way
  (the 90% band spans roughly median/2.3 to median*2.3).  In the growth
  regime s_fit is the fit residual, floored at 0.50 so a deceptively clean
  fit never claims more certainty than the model has.  A trophy at 100%
  stays at 100%: all five quantiles are 100.0.

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

# Launch-night prior, used only when players_tracked is flat (stale page):
# tracked players roughly double every 2 hours in the opening night.
DOUBLING_HOURS = 2.0
# Cap on the stale-regime players multiplier (4 doublings).
MAX_G = 12.0
# Logit spread (std) in the stale regime: honest width for a refresh whose
# size is uncertain within about a factor of two either way.
S_STALE = 0.50
MIN_FIT_RESID = 0.50
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
    ps = [p for _, p, _ in reg]
    players_flat = len(set(ps)) < 2

    # --- players model ---
    if not players_flat and len(xs) >= 2:
        a_p, b_p, _ = linfit(xs, [math.log(p) for p in ps])
        p_target = max(math.exp(a_p + b_p * x_target), ps[-1], 1.0)
    else:
        # STALE regime: exponential launch prior over the staleness window
        # (first read -> target), because the displayed count is that old.
        staleness_h = (target_t - reads[0][0]).total_seconds() / 3600.0
        staleness_h = max(staleness_h, 0.0)
        g = 2.0 ** (staleness_h / DOUBLING_HOURS)
        g = clamp(g, 1.0, MAX_G)
        p_target = ps[-1] * g
    g_mult = p_target / ps[-1] if ps[-1] > 0 else 1.0

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
        frac = clamp(last_rate, 0.01, 99.99) / 100.0
        if len(pts) >= 2 and len(set(y for _, y in pts)) >= 2:
            a_j, b_j, resid = linfit([x for x, _ in pts],
                                     [y for _, y in pts])
            s_fit = clamp(resid, MIN_FIT_RESID, MAX_FIT_RESID)
            log_e_target = a_j + b_j * x_target
            med = 100.0 * math.exp(log_e_target) / p_target
        else:
            # Flat earners: a newcomer earns this trophy with probability
            # ~ its current rate (a 100% trophy is earned by everyone, a
            # rare one by almost no newcomer), so the earners multiplier
            # is 1 + (G-1)*frac where G is the players multiplier.
            e_last = frac * ps[-1]
            e_target = e_last * (1.0 + (g_mult - 1.0) * frac)
            med = 100.0 * e_target / p_target
            s_fit = S_STALE if players_flat else MIN_FIT_RESID
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

    # A trophy at 100% stays at 100%: every newcomer earns it, so all
    # five quantiles are 100.0 when the last rate is 100 and the model
    # agrees it stays there.
    last_rates = {}
    for _, _, rates in reversed(reads):
        for j, tr in enumerate(trophies):
            if tr["id"] not in last_rates and j < len(rates) \
                    and rates[j] is not None:
                last_rates[tr["id"]] = float(rates[j])
    out = []
    for tr in trophies:
        q = rows.get(tr["id"])
        if q is None:
            q = [0.5, 0.5, 0.5, 0.5, 0.5]
        if last_rates.get(tr["id"], 0.0) >= 99.99 and q[2] >= 99.9:
            q = [100.0, 100.0, 100.0, 100.0, 100.0]
        q = [round(clamp(v, 0.01, 100.0), 4) for v in q]
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
