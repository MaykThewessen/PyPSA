# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Timezone-aware snapshots give the same results as naive UTC snapshots."""

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import pypsa

# Spans the spring DST change in Europe/Amsterdam (2020-03-29 02:00 CET -> 03:00 CEST)
UTC_HOURS = pd.date_range("2020-03-28 22:00", periods=8, freq="h")
TIMEZONES = ["UTC", "Europe/Amsterdam"]


def build(snapshots):
    n = pypsa.Network(snapshots=snapshots)
    n.add("Carrier", ["gas", "solar", "battery"])
    n.add("Bus", "bus")
    profile = np.resize([0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0], len(n.snapshots))
    n.add("Load", "load", bus="bus", p_set=3 + 3 * profile)
    n.add("Generator", "gas", bus="bus", carrier="gas", p_nom=10, marginal_cost=50)
    n.add(
        "Generator",
        "solar",
        bus="bus",
        carrier="solar",
        p_nom=8,
        marginal_cost=1,
        p_max_pu=profile,
    )
    n.add(
        "StorageUnit",
        "battery",
        bus="bus",
        carrier="battery",
        p_nom=2,
        max_hours=2,
        cyclic_state_of_charge=True,
    )
    return n


def solve(n, rolling=False):
    if rolling:
        n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=2)
    else:
        n.optimize()
    return n


@pytest.fixture(scope="module", params=TIMEZONES)
def solved(request):
    naive = solve(build(UTC_HOURS))
    aware = solve(build(UTC_HOURS.tz_localize("UTC").tz_convert(request.param)))
    return naive, aware


@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("tz", TIMEZONES)
def test_optimize_matches_naive_utc(tz, rolling):
    naive = solve(build(UTC_HOURS), rolling)
    aware = solve(build(UTC_HOURS.tz_localize("UTC").tz_convert(tz)), rolling)

    assert str(aware.snapshots.tz) == tz
    for attr in ["generators_t.p", "storage_units_t.state_of_charge"]:
        list_name, key = attr.split(".")
        result = getattr(aware, list_name)[key]
        assert result.index.equals(aware.snapshots)
        np.testing.assert_allclose(result, getattr(naive, list_name)[key])
    np.testing.assert_allclose(
        aware.buses_t.marginal_price, naive.buses_t.marginal_price
    )


def test_statistics_match_naive_utc(solved):
    naive, aware = solved

    pd.testing.assert_frame_equal(aware.statistics(), naive.statistics())
    balance = aware.statistics.energy_balance(groupby_time=False)
    assert balance.columns.equals(aware.snapshots)
    np.testing.assert_allclose(
        balance, naive.statistics.energy_balance(groupby_time=False)
    )


@pytest.mark.parametrize("multi_invest", [False, True])
@pytest.mark.parametrize(
    "fmt",
    [
        "nc",
        pytest.param(
            "h5",
            marks=pytest.mark.skipif(not find_spec("tables"), reason="no pytables"),
        ),
        "csv",
        pytest.param(
            "xlsx",
            marks=pytest.mark.skipif(
                not (find_spec("openpyxl") and find_spec("python_calamine")),
                reason="no excel dependencies",
            ),
        ),
    ],
)
def test_io_roundtrip_keeps_timezone(fmt, multi_invest, tmp_path):
    snapshots = UTC_HOURS.tz_localize("UTC").tz_convert("Europe/Amsterdam")
    if multi_invest:
        snapshots = pd.MultiIndex.from_product([[2030, 2040], snapshots])
    n = build(snapshots)
    if multi_invest:
        n.investment_periods = [2030, 2040]
    n.optimize(multi_investment_periods=multi_invest)

    path = tmp_path / ("csv_folder" if fmt == "csv" else f"network.{fmt}")
    {
        "nc": n.export_to_netcdf,
        "h5": n.export_to_hdf5,
        "csv": n.export_to_csv_folder,
        "xlsx": n.export_to_excel,
    }[fmt](path)
    m = pypsa.Network(path)

    assert str(m.timesteps.tz) == "Europe/Amsterdam"
    assert m.snapshots.equals(n.snapshots)
    # netcdf reads back datetime64[ns], other formats keep [us]
    pd.testing.assert_frame_equal(
        m.generators_t.p, n.generators_t.p, check_index_type=False, check_freq=False
    )


def test_weightings_from_timedelta_across_dst():
    n = pypsa.Network()
    n.set_snapshots(
        pd.date_range("2020-03-28", periods=3, freq="D", tz="Europe/Amsterdam"),
        weightings_from_timedelta=True,
    )
    # 29 March 2020 has 23 hours in Europe/Amsterdam
    assert n.snapshot_weightings.objective.tolist() == [24.0, 23.0, 23.0]
