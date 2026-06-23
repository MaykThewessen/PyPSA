# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Lines.apply_seasonal_rating."""

import numpy as np
import pandas as pd
import pytest

import pypsa

NH_SUMMER = (4, 5, 6, 7, 8, 9)


@pytest.fixture
def seasonal_network():
    """Single-line network with a yearly hourly index and s_nom=1000."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=8760, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=1000)
    return n


def _summer(s):
    return s[s.index.month.isin(NH_SUMMER)]


def _winter(s):
    return s[~s.index.month.isin(NH_SUMMER)]


@pytest.mark.parametrize(
    ("summer", "winter", "exp_summer_pu", "exp_winter_pu"),
    [
        (800, 1000, 0.8, 1.0),  # summer derated (typical)
        (1000, 800, 1.0, 0.8),  # winter lower than summer (data-driven)
        (1500, 1500, 1.5, 1.5),  # equal ratings, both above s_nom
        (1000, 1000, 1.0, 1.0),  # equal ratings at s_nom
    ],
)
def test_seasonal_factor_relative_to_s_nom(
    seasonal_network, summer, winter, exp_summer_pu, exp_winter_pu
):
    """s_max_pu = rating / s_nom per season; s_nom is left untouched."""
    n = seasonal_network
    ratings = pd.DataFrame({"summer": [summer], "winter": [winter]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings)

    # s_nom must not be modified in place.
    assert float(n.lines.at["a-b", "s_nom"]) == 1000.0

    s = n.lines_t.s_max_pu["a-b"]
    assert np.allclose(_summer(s), exp_summer_pu)
    assert np.allclose(_winter(s), exp_winter_pu)


def test_s_nom_never_changes_for_any_input(seasonal_network):
    """Even when ratings differ wildly from s_nom, s_nom is preserved."""
    n = seasonal_network
    before = float(n.lines.at["a-b", "s_nom"])
    ratings = pd.DataFrame({"summer": [400], "winter": [2500]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings)
    assert float(n.lines.at["a-b", "s_nom"]) == before


def test_southern_hemisphere(seasonal_network):
    """summer_months override flips which snapshots get the summer rating."""
    n = seasonal_network
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings, summer_months=(10, 11, 12, 1, 2, 3))

    s = n.lines_t.s_max_pu["a-b"]
    nh = s[s.index.month.isin(NH_SUMMER)]
    sh = s[s.index.month.isin((10, 11, 12, 1, 2, 3))]
    assert np.allclose(nh, 1.0)  # NH months now get the winter rating
    assert np.allclose(sh, 0.8)  # SH summer months get the summer rating


def test_compose_multiplies_with_static_s_max_pu(seasonal_network):
    """compose=True preserves a pre-existing static N-1 margin via multiplication."""
    n = seasonal_network
    n.lines.at["a-b", "s_max_pu"] = 0.9
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings, compose=True)

    s = n.lines_t.s_max_pu["a-b"]
    assert np.allclose(_summer(s), 0.72)  # 0.9 * 0.8
    assert np.allclose(_winter(s), 0.9)  # 0.9 * 1.0


def test_compose_false_overwrites(seasonal_network):
    """compose=False ignores any pre-existing s_max_pu."""
    n = seasonal_network
    n.lines.at["a-b", "s_max_pu"] = 0.9
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings, compose=False)

    s = n.lines_t.s_max_pu["a-b"]
    assert np.allclose(_summer(s), 0.8)
    assert np.allclose(_winter(s), 1.0)


def test_compose_with_existing_dynamic_series(seasonal_network):
    """A pre-existing per-snapshot s_max_pu composes element-wise."""
    n = seasonal_network
    pre = pd.Series(np.where(n.snapshots.hour < 12, 0.9, 0.8), index=n.snapshots)
    n.lines_t.s_max_pu["a-b"] = pre
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings, compose=True)

    s = n.lines_t.s_max_pu["a-b"]
    summer_mask = s.index.month.isin(NH_SUMMER)
    morning_summer = s[(s.index.hour < 12) & summer_mask]
    afternoon_winter = s[(s.index.hour >= 12) & ~summer_mask]
    assert np.allclose(morning_summer, 0.72)  # 0.9 * 0.8
    assert np.allclose(afternoon_winter, 0.8)  # 0.8 * 1.0


def test_multi_line_partial_subset(seasonal_network):
    """Lines absent from `ratings` are left entirely untouched."""
    n = seasonal_network
    n.add("Bus", "c")
    n.add("Line", "b-c", bus0="b", bus1="c", x=0.1, s_nom=500)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings)

    assert float(n.lines.at["a-b", "s_nom"]) == 1000.0
    assert float(n.lines.at["b-c", "s_nom"]) == 500.0
    assert "b-c" not in n.lines_t.s_max_pu.columns


def test_rejects_non_datetime_snapshots():
    n = pypsa.Network()
    n.set_snapshots(range(24))
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, s_nom=1000)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        n.c.lines.apply_seasonal_rating(ratings)


def test_rejects_missing_columns(seasonal_network):
    bad = pd.DataFrame({"summer": [800]}, index=["a-b"])
    with pytest.raises(KeyError, match="winter"):
        seasonal_network.c.lines.apply_seasonal_rating(bad)


def test_rejects_unknown_lines(seasonal_network):
    bad = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["nope"])
    with pytest.raises(KeyError, match="not in n.lines.index"):
        seasonal_network.c.lines.apply_seasonal_rating(bad)


def test_rejects_nonpositive_ratings(seasonal_network):
    bad = pd.DataFrame({"summer": [0], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="positive"):
        seasonal_network.c.lines.apply_seasonal_rating(bad)


def test_rejects_nonpositive_s_nom(seasonal_network):
    n = seasonal_network
    n.lines.at["a-b", "s_nom"] = 0.0
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="s_nom"):
        n.c.lines.apply_seasonal_rating(ratings)


def test_rejects_empty_summer_months(seasonal_network):
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="non-empty"):
        seasonal_network.c.lines.apply_seasonal_rating(ratings, summer_months=())


def test_solve_runs():
    """End-to-end: a network with seasonal ratings still solves to optimality."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=48, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=1000)
    n.add("Generator", "g", bus="a", p_nom=500, marginal_cost=10)
    n.add("Load", "ld", bus="b", p_set=400)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings)
    status, condition = n.optimize()
    assert status == "ok"
    assert condition == "optimal"
