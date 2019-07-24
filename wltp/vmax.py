#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import logging
from collections import namedtuple
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import interpolate, optimize

from .utils import make_xy_df


log = logging.getLogger(__name__)

#: Solution results of the equation finding the v-max of each gear:
#:   - v_max, p_max: in kmh, kW, `Nan` if not found
#:   - optimize_result: the ``scipy.optimize.root`` result structure
#:   - wot_df: intermediate curves for solving the equation
GearVMaxRec = namedtuple("GearVMaxRec", "v_max  p_max  optimize_result  wot_df")
#: Global resulta for v-max.
#:   - v_max, p_max: in kmh, kW, `Nan` if not found
#:   - gears_df: intermediate scalars related to each gear (scalar-item x gear)
#:   - wots_df: intermediate curves for solving each gear equation (N x (gear, curve))
#:     with mult-indexed columns.  Gear-0 contains the common columns.
VMaxRec = namedtuple("VMaxRec", "v_max  p_max  gears_df  wots_df")


def _find_p_remain_root(pv: pd.DataFrame, initial_guess) -> optimize.RootResults:
    """
    Find the velocity (the "x") where power (the "y") gets to zero.

    :param pv: 
        a 2-column dataframe for velocity & power, in that order.
    :return:
        optimization result (structure)
    """
    Spline = interpolate.InterpolatedUnivariateSpline
    # extrapolate_type: 0(extend), 1(zero), 3(boundary value)
    extrapolate_bounds = 0
    derivative_extrapolate_bounds = 3
    rank = 1

    V, P = pv.iloc[:, 0], pv.iloc[:, 1]
    pv_curve = Spline(V, P, k=rank, ext=extrapolate_bounds)
    P_grad1 = np.gradient(P)
    P_grad2 = np.gradient(P_grad1)
    pv_jacobian1 = Spline(V, P_grad1, k=rank, ext=derivative_extrapolate_bounds)

    ## NOTE: the default 'hybr' method fails to find any root!
    res = optimize.root_scalar(
        pv_curve,
        bracket=[V.min(), V.max()],
        x0=initial_guess,
        fprime=pv_jacobian1,
        method="newton",
        # Low tol because GTR requires 1 decimal point in V.
        xtol=0.0001,
    )

    return None, res


def _calc_gear_v_max(g, df: pd.DataFrame, c_p_avail, n2v, f0, f1, f2) -> GearVMaxRec:
    """
    The `v_max` for a gear `g` is the solution of :math:`0.1 * P_{avail}(g) = P_{road_loads}`.

    :param df:
        A dataframe containing at least `c_p_avail` column in kW,
        indexed by N in min^-1.
        NOTE: the power must already have been **reduced** by safety-margin,
        
        .. attention:: it appends columns in this dataframe.
        
    :param n2v:
        The n/v ratio as defined in Annex 1-2 for the gear to 
        calc its `v_max` (if it exists). 
    :return:
        a :class:`GearVMaxRec` namedtuple.

    """
    from . import formulae

    df["v"] = df.index / n2v
    df["p_road_loads"] = formulae.calc_road_load_power(df["v"], f0, f1, f2)
    df["p_remain"] = df[c_p_avail] - df["p_road_loads"]
    initial_guess_v = df.index.max() / n2v
    p_max, res = _find_p_remain_root(df[["v", "p_remain"]], initial_guess_v)

    v_max = np.NAN
    if res.converged:
        n = res.root * n2v
        v_rounded = formulae.round1(res.root)
        n_rounded = v_rounded * n2v
        if df.index.min() <= n_rounded <= df.index.max():
            v_max = res.root
            ## Interpolate solved `n` in index.
            #
            df.loc[n, :] = np.NAN
            df.loc[n_rounded, :] = np.NAN
            df.interpolate()
        elif n_rounded > df.index.max():
            v_max = df.index.max() / n2v
            # Do not interpolate solved `n`,
            # it will extend wot beyond `pwot_n_max`.

    return GearVMaxRec(v_max, p_max, res, df)


def calc_v_max(
    Pwots: Union[pd.Series, pd.DataFrame], gear_n2v_ratios, f0, f1, f2, f_safety_margin
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

    def _drop_maxv_common_columns(dfs):
        for df in dfs:
            df.drop(columns=[c_p_wot, c_p_avail], inplace=True)

    def _package_gears_df(v_maxes, p_maxes, optimize_results):
        """note: each arg is a list of items"""
        items1 = pd.DataFrame.from_dict({"v_max": v_maxes, "p_max": p_maxes})
        items2 = pd.DataFrame.from_records(vars(o) for o in optimize_results)[
            ["converged", "flag", "iterations"]
        ]
        items2.columns = "solver_ok  solver_msg  solver_nit".split()
        return pd.concat((items1, items2), axis=1)

    def _package_wots_df(wot_df, solution_dfs):
        _drop_maxv_common_columns(solution_dfs)
        wots_df = pd.concat(solution_dfs, axis=1, keys=range(0, ng + 1))
        wot_df[c_n] = wot_df.index
        ###wots_df.index = wot_df.values

        return wots_df

    wot_df = make_xy_df(Pwots, xname=c_n, yname=c_p_wot, auto_transpose=True)
    wot_df[c_p_avail] = wot_df[c_p_wot] * (1.0 - f_safety_margin)

    recs: List[GearVMaxRec] = []
    for g, n2v in reversed(list(enumerate(gear_n2v_ratios, 1))):
        rec = _calc_gear_v_max(g, wot_df.copy(), c_p_avail, n2v, f0, f1, f2)
        if not recs or recs[-1].v_max < rec.v_max:
            recs.append(rec)

    *gears_infos, wot_solution_dfs = zip(*recs)

    gears_df = _package_gears_df(*gears_infos)
    wots_df = _package_wots_df(wot_df, wot_solution_dfs)
    v_max = gears_df["v_max"].max()
    p_max = gears_df["p_max"].max()

    return VMaxRec(v_max, p_max, gears_df, wots_df)
