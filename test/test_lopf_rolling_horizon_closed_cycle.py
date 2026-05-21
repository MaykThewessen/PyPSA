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


def test_default_path_unchanged() -> None:
    """Without pin_terminal_soc, behaviour matches today's rolling horizon."""
    n_a = _build_network_with_storage()
    n_b = _build_network_with_storage()

    n_a.optimize.optimize_with_rolling_horizon(horizon=6)
    n_b.optimize.optimize_with_rolling_horizon(horizon=6, pin_terminal_soc=False)

    pd.testing.assert_frame_equal(
        n_a.generators_t.p, n_b.generators_t.p, check_dtype=False
    )


def test_pin_terminal_soc_holds_for_every_chunk() -> None:
    """Terminal SoC of every chunk equals boundary_soc * capacity."""
    n = _build_network_with_storage()
    horizon = 6
    n.optimize.optimize_with_rolling_horizon(
        horizon=horizon, pin_terminal_soc=True, boundary_soc=0.5
    )

    cap_su = (
        n.c.storage_units.static.loc["battery", "p_nom"]
        * (n.c.storage_units.static.loc["battery", "max_hours"])
    )
    cap_store = n.c.stores.static.loc["tank", "e_nom"]

    chunk_ends = list(range(horizon - 1, len(n.snapshots), horizon))
    for end in chunk_ends:
        sn = n.snapshots[end]
        assert n.storage_units_t.state_of_charge.loc[sn, "battery"] == pytest.approx(
            0.5 * cap_su, abs=1e-2
        )
        assert n.stores_t.e.loc[sn, "tank"] == pytest.approx(0.5 * cap_store, abs=1e-2)


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


def test_cyclic_flags_restored_on_success() -> None:
    n = _build_network_with_storage(cyclic_su=True, cyclic_store=True)
    n.optimize.optimize_with_rolling_horizon(
        horizon=6, pin_terminal_soc=True, boundary_soc=0.5
    )
    assert n.c.storage_units.static.loc["battery", "cyclic_state_of_charge"]
    assert n.c.stores.static.loc["tank", "e_cyclic"]


def test_cyclic_flags_and_pins_restored_on_failure(monkeypatch) -> None:
    """If the solver call raises mid-loop, the network must end up byte-identical.

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

    def _boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(OptimizationAccessor, "__call__", _boom)

    with pytest.raises(RuntimeError, match="simulated solver failure"):
        n.optimize.optimize_with_rolling_horizon(
            horizon=6, pin_terminal_soc=True, boundary_soc=0.5
        )

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


def test_monkeypatch_target_assumption() -> None:
    """Guard the failure-test's load-bearing assumption about the accessor.

    If someone refactors the accessor so that ``optimize_with_rolling_horizon``
    routes through ``__call__``, the failure test above silently changes
    semantics. This guard fails loudly in that case.
    """
    from pypsa.optimization.optimize import OptimizationAccessor

    assert "__call__" in OptimizationAccessor.__dict__
    found_method = any(
        "optimize_with_rolling_horizon" in c.__dict__
        for c in OptimizationAccessor.__mro__
    )
    assert found_method


def test_boundary_soc_as_mapping() -> None:
    n = _build_network_with_storage()
    n.optimize.optimize_with_rolling_horizon(
        horizon=6,
        pin_terminal_soc=True,
        boundary_soc={"battery": 0.75, "tank": 0.25},
    )

    cap_su = (
        n.c.storage_units.static.loc["battery", "p_nom"]
        * n.c.storage_units.static.loc["battery", "max_hours"]
    )
    cap_store = n.c.stores.static.loc["tank", "e_nom"]

    end = n.snapshots[5]
    assert n.storage_units_t.state_of_charge.loc[end, "battery"] == pytest.approx(
        0.75 * cap_su, abs=1e-2
    )
    assert n.stores_t.e.loc[end, "tank"] == pytest.approx(0.25 * cap_store, abs=1e-2)


def test_boundary_soc_as_callable() -> None:
    n = _build_network_with_storage()

    def policy(_network: pypsa.Network, snapshot: int) -> float:
        idx = int(snapshot)
        return 0.4 + 0.2 * idx / 23.0

    n.optimize.optimize_with_rolling_horizon(
        horizon=8, pin_terminal_soc=True, boundary_soc=policy
    )

    cap_su = 50 * 4
    for end_idx in (7, 15, 23):
        sn = n.snapshots[end_idx]
        soc = n.storage_units_t.state_of_charge.loc[sn, "battery"]
        assert 0.4 * cap_su - 1e-2 <= soc <= 0.6 * cap_su + 1e-2


def test_boundary_soc_out_of_range_rejected() -> None:
    n = _build_network_with_storage()
    with pytest.raises(ValueError, match=r"boundary_soc must lie in \[0, 1\]"):
        n.optimize.optimize_with_rolling_horizon(
            horizon=6, pin_terminal_soc=True, boundary_soc=1.5
        )


def test_boundary_soc_bad_type_rejected() -> None:
    n = _build_network_with_storage()
    with pytest.raises(TypeError, match="boundary_soc must be"):
        n.optimize.optimize_with_rolling_horizon(
            horizon=6, pin_terminal_soc=True, boundary_soc="half"
        )


def test_storage_unit_only_network() -> None:
    """No Stores in the network; pin only StorageUnit terminals."""
    n = _build_network_with_storage()
    n.remove("Store", "tank")
    n.optimize.optimize_with_rolling_horizon(
        horizon=6, pin_terminal_soc=True, boundary_soc=0.5
    )
    assert n.objective > 0


def test_store_only_network() -> None:
    """No StorageUnits in the network; pin only Store terminals."""
    n = _build_network_with_storage()
    n.remove("StorageUnit", "battery")
    n.optimize.optimize_with_rolling_horizon(
        horizon=6, pin_terminal_soc=True, boundary_soc=0.5
    )
    assert n.objective > 0
