# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BGBug22/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BGBug22/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Forecast program for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01): it reads a case file describing the 55 GTA VI PS5 trophies and their Exophase earn-rate history, then writes 5 quantiles (5%, 25%, 50%, 75%, 95%) per trophy for the target date 26 Nov 2026 00:00 UTC. Scored by mean pinball loss on the logit scale; the prize goes to entries beating the best house entry.

Built for the Neon Coastal Open-World Trophy & Achievement Showcase theme.

## How it forecasts

`main.py` (Python 3.13, stdlib only, no network) reads every `*.in` case in
the input dir and writes one `<id>.out` per case. For each trophy it models
`rate = earners / players_tracked`: a power law in hours since release is
fitted to the players count and to each trophy's earner count over the
current regime (after the last big refresh jump, else the last 24 h), both
are predicted at the target, and the rate is their ratio. When a series is
flat (no growth signal, as on launch night) a launch-surge prior is used
instead of pretending nothing changes: players keep growing while earners
grow at half that log-rate, so rates fall — the direction a launch refresh
takes them. Quantile spread lives on the logit scale, combining a
random-walk term over the horizon with the fit residual. The forecast is
then made consistent: Platinum is capped at the rarest other trophy, story
trophies (first mission, Chapters 1–4, final mission) never rise along the
chain, and long-grind / collect-everything trophies never exceed the
hardest story chapter's rate.
