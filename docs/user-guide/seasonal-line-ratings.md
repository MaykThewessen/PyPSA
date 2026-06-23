<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Seasonal Line Ratings

Transmission lines have different thermal capacities in summer and winter,
because conductor ampacity depends on how well the line sheds heat to the
surrounding air. Transmission system operators publish per-line summer and
winter ratings in **absolute capacity** terms (MVA). For example, the JAO /
Core flow-based domain files state a maximum permissible current per season and
voltage level, which together define an absolute MVA limit for the line.

The [`Lines.apply_seasonal_rating`][pypsa.components._types.lines.Lines.apply_seasonal_rating]
method takes such a per-line `(summer, winter)` MVA table and writes the
seasonal envelope into `n.lines_t.s_max_pu`. Crucially, it does **not** change
`n.lines.s_nom`: the nominal rating you built the network with stays fixed, and
the seasonal variation is expressed as a per-unit factor `rating / s_nom`
relative to it.

## When to use it

- A TSO publishes per-line summer / winter MVA ratings and you want them in the
  dispatch problem.
- You already model an N-1 margin via `n.lines.s_max_pu` and want to compose a
  seasonal factor on top of it.
- You want a small, dependency-free utility, not a JAO loader or a dynamic line
  rating (DLR) engine. The caller assembles the `ratings` table.

## Minimal example

```python
import pandas as pd
import pypsa

n = pypsa.Network()
n.set_snapshots(pd.date_range("2025-01-01", periods=8760, freq="h"))
n.add("Bus", ["a", "b"])
n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, s_nom=1000)

ratings = pd.DataFrame(
    {"summer": [800], "winter": [1000]},  # absolute MVA per season
    index=["a-b"],
)
n.c.lines.apply_seasonal_rating(ratings)

# s_nom is unchanged; the seasonal limit lives in s_max_pu.
float(n.lines.at["a-b", "s_nom"])         # 1000.0
n.lines_t.s_max_pu["a-b"].iloc[3000]      # 0.8 (mid-summer hour, 800 / 1000)
```

For each snapshot the factor written to `s_max_pu` is `rating(season) / s_nom`:
the summer rating on summer months and the winter rating elsewhere. Ratings
above `s_nom` give a factor greater than 1.

## Composing with an N-1 margin

By default the method multiplies the seasonal factor into any pre-existing
`s_max_pu` so that margins such as N-1 survive:

```python
n.lines.at["a-b", "s_max_pu"] = 0.9   # static N-1 margin
n.c.lines.apply_seasonal_rating(ratings, compose=True)

# Summer hour:  0.9 * 0.8 = 0.72
# Winter hour:  0.9 * 1.0 = 0.9
```

Pre-existing dynamic series in `n.lines_t.s_max_pu` compose element-wise. Pass
`compose=False` to overwrite instead.

## Southern hemisphere

The default `summer_months=(4, 5, 6, 7, 8, 9)` is northern-hemisphere. Override
for SH networks:

```python
n.c.lines.apply_seasonal_rating(
    ratings, summer_months=(10, 11, 12, 1, 2, 3),
)
```

## Raises

- `TypeError` if `n.snapshots` is not a `DatetimeIndex`.
- `KeyError` if `ratings` references a line not in `n.lines.index`, or is
  missing the `summer` / `winter` columns.
- `ValueError` if any rating is non-positive, if `summer_months` is empty, or if
  a referenced line has a non-positive `s_nom`.
