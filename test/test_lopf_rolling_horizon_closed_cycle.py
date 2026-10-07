# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for closed-cycle rolling horizon (``pin_terminal_soc``)."""

from __future__ import annotations

import pandas as pd
import pytest

import pypsa


def _build_network_with_storage(
    cyclic_su: bool = False, cyclic_store: bool = False
) -> pypsa.Network:
    """Build a 24h single-bus network with one StorageUnit and one Store.

    The cheap-then-expensive marginal-cost profile gives the LP a clear
    incentive to drain storage by the end of the horizon, which is what the
    closed-cycle pin must defeat.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(24))
    n.add("Bus", "bus")

    n.add("Generator", "cheap", bus="bus", p_nom=100, marginal_cost=5)
    n.add("Generator", "slack", bus="bus", p_nom=1000, marginal_cost=1000)

    cost = pd.DataFrame({"cheap": [5.0] * 12 + [200.0] * 12}, index=n.snapshots)
    n.generators_t.marginal_cost = cost
    n.add("Load", "load", bus="bus", p_set=80)

    n.add(
        "StorageUnit",
        "battery",
        bus="bus",
        p_nom=50,
        max_hours=4,
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        state_of_charge_initial=100.0,
        cyclic_state_of_charge=cyclic_su,
    )
    n.add(
        "Store",
        "tank",
        bus="bus",
        e_nom=200,
        e_initial=100.0,
        e_cyclic=cyclic_store,
    )
    return n


@pytest.mark.parametrize("drop", [None, "Store", "StorageUnit"])
@pytest.mark.parametrize(
    ("horizon", "overlap", "seams"),
    [
        (6, 0, [5, 11, 17, 23]),
        (8, 3, [4, 9, 14, 19, 23]),
    ],
)
def test_pin_holds_at_window_seams(
    horizon: int, overlap: int, seams: list[int], drop: str | None
) -> None:
    """SoC equals boundary_soc * capacity at each window's last committed snapshot."""
    n = _build_network_with_storage()
    if drop == "Store":
        n.remove("Store", "tank")
    elif drop == "StorageUnit":
        n.remove("StorageUnit", "battery")

    n.optimize.optimize_with_rolling_horizon(
        horizon=horizon, overlap=overlap, pin_terminal_soc=True, boundary_soc=0.5
    )

    for sn in seams:
        if drop != "StorageUnit":
            assert n.storage_units_t.state_of_charge.loc[
                sn, "battery"
            ] == pytest.approx(0.5 * 50 * 4, abs=1e-2)
        if drop != "Store":
            assert n.stores_t.e.loc[sn, "tank"] == pytest.approx(0.5 * 200, abs=1e-2)


def test_pin_prevents_end_of_chunk_drain() -> None:
    """Pinning forces a higher terminal SoC than the open-ended rolling solve."""
    n_open = _build_network_with_storage()
    n_pin = _build_network_with_storage()

    n_open.optimize.optimize_with_rolling_horizon(horizon=12)
    n_pin.optimize.optimize_with_rolling_horizon(
        horizon=12, pin_terminal_soc=True, boundary_soc=0.5
    )

    soc_open = n_open.storage_units_t.state_of_charge.iloc[-1]["battery"]
    soc_pin = n_pin.storage_units_t.state_of_charge.iloc[-1]["battery"]
    assert soc_pin > soc_open


def test_user_set_points_survive() -> None:
    """Set points the user gave for a subset of units are honoured and restored."""
    n = _build_network_with_storage()
    n.add(
        "StorageUnit",
        "battery2",
        bus="bus",
        p_nom=50,
        max_hours=4,
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        state_of_charge_initial=100.0,
    )
    soc_set = pd.Series(float("nan"), index=n.snapshots)
    soc_set.loc[2] = 60.0
    n.storage_units_t.state_of_charge_set["battery"] = soc_set
    su_set_before = n.storage_units_t.state_of_charge_set.copy()

    n.optimize.optimize_with_rolling_horizon(
        horizon=6, pin_terminal_soc=True, boundary_soc=0.5
    )

    assert n.storage_units_t.state_of_charge.loc[2, "battery"] == pytest.approx(
        60.0, abs=1e-2
    )
    pd.testing.assert_frame_equal(
        n.storage_units_t.state_of_charge_set, su_set_before, check_dtype=False
    )


@pytest.mark.parametrize("fail", [False, True])
def test_storage_state_restored(monkeypatch, fail: bool) -> None:  # noqa: ANN001
    """Cyclic flags and set-point frames are restored after success and failure.

    ``n.optimize(...)`` dispatches through ``OptimizationAccessor.__call__``.
    Python's special-method lookup goes via the type, not the instance, so we
    patch the class. ``optimize_with_rolling_horizon`` is a regular bound
    method and is unaffected.
    """
    from pypsa.optimization.optimize import OptimizationAccessor

    n = _build_network_with_storage(cyclic_su=True, cyclic_store=True)

    su_cyclic_before = n.c.storage_units.static["cyclic_state_of_charge"].copy()
    st_cyclic_before = n.c.stores.static["e_cyclic"].copy()
    su_set_before = n.c.storage_units.dynamic.state_of_charge_set.copy()
    st_set_before = n.c.stores.dynamic.e_set.copy()

    def run() -> None:
        n.optimize.optimize_with_rolling_horizon(
            horizon=6, pin_terminal_soc=True, boundary_soc=0.5
        )

    if fail:

        def _boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("simulated solver failure")

        monkeypatch.setattr(OptimizationAccessor, "__call__", _boom)
        with pytest.raises(RuntimeError, match="simulated solver failure"):
            run()
    else:
        run()

    pd.testing.assert_series_equal(
        n.c.storage_units.static["cyclic_state_of_charge"],
        su_cyclic_before,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        n.c.stores.static["e_cyclic"], st_cyclic_before, check_names=False
    )
    pd.testing.assert_frame_equal(
        n.c.storage_units.dynamic.state_of_charge_set,
        su_set_before,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        n.c.stores.dynamic.e_set, st_set_before, check_dtype=False
    )


@pytest.mark.parametrize("boundary_soc", [-0.1, 1.5])
def test_boundary_soc_out_of_range_rejected(boundary_soc: float) -> None:
    n = _build_network_with_storage()
    with pytest.raises(ValueError, match=r"boundary_soc must lie in \[0, 1\]"):
        n.optimize.optimize_with_rolling_horizon(
            horizon=6, pin_terminal_soc=True, boundary_soc=boundary_soc
        )
