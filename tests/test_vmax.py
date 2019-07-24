#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import logging
import random

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from wltp import formulae, vmax

from . import nbutils as nbu

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def test_v_max(h5db):
    from . import conftest

    nsamples = 110

    def make_v_maxes(vehnum):
        iprops, Pwot, n2vs = conftest._load_vehicle_data(h5db, vehnum)
        rec = vmax.calc_v_max(
            Pwot["Pwot"], n2vs, iprops.f0, iprops.f1, iprops.f2, f_safety_margin=0.10
        )
        v_max_calced = rec.v_max
        v_max_round = formulae.round1(v_max_calced, 1)
        v_max_heinz = iprops["v_max"]
        return v_max_calced, v_max_round, v_max_heinz, rec.gears_df

    veh_nums = nbu.all_vehnums(h5db)
    recs = np.array(
        [
            make_v_maxes(vehnum)
            for vehnum in random.sample(veh_nums, nsamples or len(veh_nums))
        ]
    )
    v_maxes_calced, v_maxes_round, v_maxes_heinz, gears_dfs = recs.T
    gears_df = pd.concat(gears_dfs, keys=range(len(gears_dfs)))
    print(
        "iterations_count(ok):",
        gears_df.loc[gears_df.solver_ok, "solver_nit"].describe(),
    )
    print(
        f"v_max diffs: {pd.Series((v_maxes_calced - v_maxes_heinz)).describe()}"
        f",\n  nones: {np.isnan(v_maxes_calced.astype('float64')).sum()} (out of {len(veh_nums)})"
    )
    npt.assert_array_equal(v_maxes_round, v_maxes_heinz)
