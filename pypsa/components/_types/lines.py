# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Lines components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.geo import haversine_pts

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr


#: Default months treated as summer for northern-hemisphere networks.
_DEFAULT_SUMMER_MONTHS_NH: tuple[int, ...] = (4, 5, 6, 7, 8, 9)


@patch_add_docstring
class Lines(Components):
    """Lines components class.

    This class is used for line components. All functionality specific to
    lines is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.lines
    'Line' Components
    -----------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 7

    """

    _operational_variables = ["s"]

    def get_bounds_pu(
        self,
        attr: str = "s",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for lines.

        <!-- md:badge-version v1.0.0 -->

        For passive branch components, min_pu is the negative of max_pu.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "s"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For lines those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        max_pu = self.da.s_max_pu
        min_pu = -max_pu  # Lines specific: min_pu is the negative of max_pu

        return min_pu, max_pu

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )

    def apply_seasonal_rating(
        self,
        ratings: pd.DataFrame,
        summer_months: Sequence[int] = _DEFAULT_SUMMER_MONTHS_NH,
        *,
        compose: bool = True,
    ) -> None:
        """Apply seasonal line ratings via ``s_max_pu`` without touching ``s_nom``.

        Transmission system operators publish per-line summer and winter line
        ratings in **absolute capacity** terms (MVA), not as per-unit load
        factors. They are derived from the seasonal ampacity of the conductor:
        for example the JAO / Core flow-based domain files state a maximum
        permissible current per season and voltage level, which together define
        an absolute MVA limit for the line.

        This helper converts those absolute seasonal ratings into per-unit
        scaling factors relative to each line's existing ``s_nom`` and writes
        them to ``n.lines_t.s_max_pu``. ``s_nom`` is **never modified in place**:
        it remains the nominal rating the network was built with, so capacity
        and investment logic that depends on it is unaffected. For each snapshot
        the factor applied is ``rating(season) / s_nom``.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Index: line names (must be a subset of ``n.lines.index``).
            Columns: ``'summer'`` and ``'winter'`` (case-sensitive), per-line
            ratings in **absolute MVA** (not per-unit).
        summer_months : Sequence[int], default ``(4, 5, 6, 7, 8, 9)``
            Snapshot months treated as summer. Default is northern-hemisphere;
            for southern-hemisphere use, pass e.g. ``(10, 11, 12, 1, 2, 3)``.
        compose : bool, default True
            If ``True``, multiply the seasonal factor into any pre-existing
            ``s_max_pu`` (static value or dynamic series) so that margins such
            as an N-1 derating survive. If ``False``, overwrite
            ``n.lines_t.s_max_pu`` with the seasonal factor alone.

        Raises
        ------
        TypeError
            If ``n.snapshots`` is not a ``DatetimeIndex``.
        KeyError
            If ``ratings`` references a line not in ``n.lines.index``, or if
            ``ratings`` is missing the ``summer`` / ``winter`` columns.
        ValueError
            If any rating is non-positive, if ``summer_months`` is empty, or if
            any referenced line has a non-positive ``s_nom``.

        Examples
        --------
        >>> import pandas as pd, pypsa
        >>> n = pypsa.Network()
        >>> n.set_snapshots(pd.date_range('2025-01-01', periods=8760, freq='h'))
        >>> n.add('Bus', ['a', 'b'])
        >>> n.add('Line', 'a-b', bus0='a', bus1='b', x=0.1, s_nom=1000)
        >>> ratings = pd.DataFrame(
        ...     {'summer': [800], 'winter': [1000]}, index=['a-b']
        ... )
        >>> n.c.lines.apply_seasonal_rating(ratings)
        >>> float(n.lines.at['a-b', 's_nom'])  # unchanged
        1000.0
        >>> float(n.lines_t.s_max_pu['a-b'].iloc[3000])  # mid-summer hour
        0.8

        """
        n = self.n_save
        if not isinstance(n.snapshots, pd.DatetimeIndex):
            msg = (
                "apply_seasonal_rating requires n.snapshots to be a "
                f"DatetimeIndex; got {type(n.snapshots).__name__}."
            )
            raise TypeError(msg)

        missing_cols = {"summer", "winter"} - set(ratings.columns)
        if missing_cols:
            msg = f"`ratings` DataFrame is missing required columns: {sorted(missing_cols)}"
            raise KeyError(msg)

        missing_lines = ratings.index.difference(self.static.index)
        if len(missing_lines) > 0:
            msg = (
                f"`ratings` references {len(missing_lines)} line(s) not in "
                f"n.lines.index: {list(missing_lines[:5])}"
                + ("..." if len(missing_lines) > 5 else "")
            )
            raise KeyError(msg)

        if not len(summer_months):
            msg = "`summer_months` must be non-empty."
            raise ValueError(msg)

        summer = pd.to_numeric(ratings["summer"], errors="raise")
        winter = pd.to_numeric(ratings["winter"], errors="raise")
        if (summer <= 0).any() or (winter <= 0).any():
            msg = "All summer and winter ratings must be strictly positive."
            raise ValueError(msg)

        s_nom = self.static.loc[ratings.index, "s_nom"]
        if (s_nom <= 0).any():
            bad = list(s_nom.index[s_nom <= 0][:5])
            msg = (
                "Seasonal ratings scale s_max_pu relative to s_nom, which must "
                f"be strictly positive; got non-positive s_nom for line(s): {bad}."
            )
            raise ValueError(msg)

        summer_pu = (summer / s_nom).to_numpy()
        winter_pu = (winter / s_nom).to_numpy()

        summer_mask = n.snapshots.month.isin(summer_months)
        seasonal = pd.DataFrame(
            np.where(summer_mask[:, None], summer_pu[None, :], winter_pu[None, :]),
            index=n.snapshots,
            columns=ratings.index,
        )

        if compose:
            existing = self.dynamic.s_max_pu.reindex(
                index=n.snapshots, columns=ratings.index
            )
            static = self.static.loc[ratings.index, "s_max_pu"]
            existing = existing.fillna(static)
            self.dynamic.s_max_pu[ratings.index] = (existing * seasonal).to_numpy()
        else:
            self.dynamic.s_max_pu[ratings.index] = seasonal.to_numpy()

    def calculate_line_length(self) -> pd.Series:
        """Get length of the lines in meters.

        Based on coordinates of attached buses. Buses must have 'x' and 'y' attributes,
        otherwise no line length can be calculated. By default the haversine formula is
        used to calculate the distance between two points.

        Returns
        -------
        pd.Series
            Length of the lines.

        See Also
        --------
        [pypsa.geo.haversine][]

        Examples
        --------
        >>> c = pypsa.examples.scigrid_de().c.lines
        >>> ds = c.calculate_line_length()
        >>> ds.head()
        0    34432.796096
        1    59701.666027
        2    32242.741010
        3    30559.154647
        4    21574.543367
        dtype: float64

        """
        return (
            pd.Series(
                haversine_pts(
                    a=np.array(
                        [
                            self.static.bus0.map(self.n_save.buses.x),
                            self.static.bus0.map(self.n_save.buses.y),
                        ]
                    ).T,
                    b=np.array(
                        [
                            self.static.bus1.map(self.n_save.buses.x),
                            self.static.bus1.map(self.n_save.buses.y),
                        ]
                    ).T,
                )
            )
            * 1_000
        )
