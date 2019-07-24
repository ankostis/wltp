#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import logging
from collections import ChainMap, namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from scipy import interpolate, optimize

from .utils import make_xy_df

log = logging.getLogger(__name__)

#: Global resulta for v-max.
#:   - v_max, p_max: in kmh, kW, `Nan` if not found
#:   - gears_df: intermediate scalars related to each gear (scalar-item x gear)
#:   - wots_df: intermediate curves for solving each gear equation (N x (gear, curve))
#:     with mult-indexed columns.  Gear-0 contains the common columns.
VMaxRec = namedtuple("VMaxRec", "v_max  p_max  gears_df  wots_df")

c_n, c_p_avail, c_p_wot = "n  p_avail  p_wot".split()
c_v, c_p_road_loads, c_p_remain = "v  p_road_loads  p_remain".split()


@dataclass
class GearVMax:
    """
    Solve :math:`P_{remain}(V) = 0` and store intermediate results to find the v-max of a gear:

    :ivar v_max: 
        (output) in kmh, `Nan` if not found
    :ivar  optimize_result: 
        the ``scipy.optimize.root`` result structure
    :wot_df: intermediate curves for solving the equation
    """

    #### INPUT ####
    #
    #: A dataframe containing at least `c_p_avail` column in kW,
    #: indexed by N in min^-1.
    #:
    #: NOTE: the power must already have been **reduced** by safety-margin,
    wot_df: pd.DataFrame
    #
    gear_n2v_ratio: float
    #
    #: A callable calculating the power of road-load resistances
    #: for a given velocity.
    pv_road_loads_func: Callable[[Any], Any]
    #
    #: For initial guess.
    n_rated: int
    #
    ###############

    ## Power interpolation
    #
    Spline = interpolate.InterpolatedUnivariateSpline
    interp_args = {
        # rank
        "k": 1,
        # extrapolate_type: 0(extend), 1(zero), 3(boundary value)
        "ext": 0,
    }
    # merged on top of `interp_args`
    derivative_interp_args = {"ext": 3}

    ## Equation solver
    #
    solver_args = {
        "method": "broyden1",
        # Low tol because GTR requires 1 decimal point in V.
        "tol": 0.001,
    }

    def _make_pv_interp(self, V, P):
        return self.Spline(V, P, **self.interp_args)

    def _make_derivative_pv_interp(self, V, P):
        derivative_intrp_args = ChainMap(self.derivative_interp_args, self.interp_args)
        return self.Spline(V, np.gradient(P), **derivative_intrp_args)

    def _make_initial_guess(self) -> float:
        return self.n_rated / self.gear_n2v_ratio

    def _find_p_remain_root(self, pv: pd.DataFrame) -> optimize.OptimizeResult:
        """
        Find the velocity (the "x") where power (the "y") gets to zero.

        :param pv: 
            a 2-column dataframe for velocity & power, in that order.
        :return:
            optimization result (structure)
        """
        V, P = pv.iloc[:, 0], pv.iloc[:, 1]
        self.pv_curve = self._make_pv_interp(V, P)
        self.pv_derivative = self._make_derivative_pv_interp(V, P)

        ## NOTE: the default 'hybr' method fails to find any root!
        return optimize.root(
            self.pv_curve,
            x0=self._make_initial_guess(),
            jac=self.pv_derivative,
            **self.solver_args
        )

    def _calc_gear_v_max(
        self, g, df: pd.DataFrame, c_p_avail, gear_n2v_ratio, f0, f1, f2
    ):
        """
        The `v_max` for a gear `g` is the solution of :math:`0.1 * P_{avail}(g) = P_{road_loads}`.

        :param df:
            A dataframe containing at least `c_p_avail` column in kW,
            indexed by N in min^-1.
            NOTE: the power must already have been **reduced** by safety-margin,
            
            .. attention:: it appends columns in this dataframe.
            
        :param gear_n2v_ratio:
            The n/v ratio as defined in Annex 1-2 for the gear to 
            calc its `v_max` (if it exists). 

        """

        df[c_v] = df.index / gear_n2v_ratio
        df[c_p_road_loads] = self.pv_road_loads_func(df[c_v])
        df[c_p_remain] = df[c_p_avail] - df[c_p_road_loads]
        res = self._find_p_remain_root(df[[c_v, c_p_remain]])
        v_max = res.x if res.success else np.NaN
        if res.success:
            log.info(
                "gear %s: , vmax: %s, p_remain: %s, nit: %s", g, v_max, p_max, res.nit
            )
        return GearVMaxRec(v_max, -1, res, df)


# TODO: get it from model
f_safety_margin = 0.1


def calc_v_max(
    Pwots: Union[pd.Series, pd.DataFrame], gear_n2v_ratios, pv_road_loads_func
) -> VMaxRec:
    """
    Finds the maximum velocity achieved by all gears.

    :param Pwots:
        A a 2D-matrix(lists/numpy), dict, df(1-or-2 cols) or series 
        containing the corresponding P(kW) value for each N in its index,
        or a 2-column matrix. 
    :param gear_n2v_ratios:
        A sequence of n/v ratios, as defined in Annex 1-2.e.
        It's length defines the number of gears to process.
    :return:
        a :class:`VMaxRec` namedtuple.
    """
    c_n, c_p_avail, c_p_wot = "n  p_avail  p_wot".split()
    ng = len(gear_n2v_ratios)

    def _drop_maxv_common_columns(self, dfs):
        for df in dfs:
            df.drop(columns=[c_p_wot, c_p_avail], inplace=True)

    def _package_gears_df(self, v_maxes, p_maxes, optimize_results):
        """note: each arg is a list of items"""
        items1 = pd.DataFrame.from_dict({"v_max": v_maxes, "p_max": p_maxes})
        items2 = pd.DataFrame.from_records(optimize_results)[
            [
                "success",
                "status",
                "message",
                "nit",
            ]  # , "nfev", "njev"]  # for other solvers
        ]
        items2.columns = "solver_ok solver_status solver_msg solver_nit".split()
        return pd.concat((items1, items2), axis=1)

    def _package_wots_df(self, wot_df, solution_dfs):
        _drop_maxv_common_columns(solution_dfs)
        wots_df = pd.concat(solution_dfs, axis=1, keys=range(0, ng + 1))
        wot_df[c_n] = wot_df.index
        wots_df.index = wot_df.values

        return wots_df

    wot_df = make_xy_df(Pwots, xname=c_n, yname=c_p_wot, auto_transpose=True)
    wot_df[c_p_avail] = wot_df[c_p_wot] * (1.0 - f_safety_margin)
    recs = [
        GearVMax(
            wot_df=wot_df.copy(),
            gear_n2v_ratio=n2v,
            pv_road_loads_func=pv_road_loads_func,
            n_rated=n_rated,
        )
        for g, n2v in enumerate(gear_n2v_ratios)
    ]

    *gears_infos, wot_solution_dfs = zip(*recs)

    gears_df = _package_gears_df(*gears_infos)
    wots_df = _package_wots_df(wot_df, wot_solution_dfs)
    v_max = gears_df["v_max"].max()
    p_max = gears_df["p_max"].max()

    return VMaxRec(v_max, p_max, gears_df, wots_df)
