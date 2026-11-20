# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BGBug22/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BGBug22/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Forecast program for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01): it reads a case file describing the 55 GTA VI PS5 trophies and their Exophase earn-rate history, then writes 5 quantiles (5%, 25%, 50%, 75%, 95%) per trophy for the target date 26 Nov 2026 00:00 UTC. Scored by mean pinball loss on the logit scale; the prize goes to entries beating the best house entry.

Built for the Neon Coastal Open-World Trophy & Achievement Showcase theme.

## How it forecasts (v2 — regime-aware)

`main.py` (Python 3.13, stdlib only, no network) reads every `*.in` case in
the input dir and writes one `<id>.out` per case. For each trophy it models
`rate = earners / players_tracked`, and the regime is detected from the
case's own history — no practice number is hardcoded anywhere:

- **Growth regime** (players_tracked moves): a power law in hours since
  release is fitted to the players count and to each trophy's earner
  count over the current regime (after the last big refresh jump, else
  the last 24 h), both are predicted at the target, and the rate is their
  ratio. This is the regime the final case (f00) lives in: its history
  already holds the post-refresh reads, so the growth is visible.
- **Stale regime** (players_tracked flat across every read — the page is
  a frozen Exophase snapshot, as on launch night): the displayed count is
  as old as the first read, so the players multiplier uses a launch-night
  exponential prior — tracked players roughly double every 2 hours in
  the opening night — applied over the staleness window (first read to
  target), capped at 12x. Earners grow only via newcomers, so a 100%
  trophy is earned by every newcomer and stays at 100%, while a rare
  one's earners barely move and its rate falls by about the players
  multiplier.

Quantile spread lives on the logit scale, combining a random-walk term
over the horizon with the fit residual: in the stale regime the spread is
an honest 0.50 (a refresh whose size is uncertain within about a factor
of two either way), and in the growth regime the residual is floored at
0.50 so a deceptively clean fit never claims more certainty than the
model has. The forecast is then made consistent: Platinum is capped at
the rarest other trophy, story trophies (first mission, Chapters 1–4,
final mission) never rise along the chain, and long-grind /
collect-everything trophies never exceed the hardest story chapter's
rate.

## Local practice check (v2 vs v1, oracle = refresh approximation)

| Case | v1 loss | v2 loss | v1 cov90 | v2 cov90 | v1 miss | v2 miss |
|------|---------|---------|----------|----------|---------|---------|
| p00  | 0.881   | 0.146   | 0.000    | 1.000    | 0.30 pp | 0.04 pp |
| p01  | 0.950   | 0.113   | 0.000    | 1.000    | 0.33 pp | 0.01 pp |

The oracle is a local test approximation only (earners ~constant, players
638 → 3,868, 100% trophy stays 100%); it never enters the model. The
real practice answer has earners growing a little, so the real loss is
expected to be lower still. The practice does not decide the prize — the
final case (f00, target 26 Nov 2026) does, and there the growth-regime
power law runs.
