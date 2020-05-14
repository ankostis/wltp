#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2013-2020 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""formulae for cycle/vehicle dynamics"""
import logging
from typing import Union

import numpy as np
import pandas as pd


from . import io as wio
from . import vehicle
from .autograph import autographed
from .invariants import Column

log = logging.getLogger(__name__)


def calc_unladen_mass(mro, driver_mass):
    return mro - driver_mass


def calc_mro(unladen_mass, driver_mass):
    return unladen_mass + driver_mass


def calc_p_m_ratio(p_rated, unladen_mass):
    return 1000 * p_rated / unladen_mass


def calc_p_resist(V: Column, f0, f1, f2):
    """
    The `p_resist` required to overcome vehicle-resistances for various velocities,

    as defined in Annex 2-2.i (calculate `V_max_vehicle`).
    """
    VV = V * V
    VVV = VV * V
    return (f0 * V + f1 * VV + f2 * VVV) / 3600.0


def attach_p_resist_in_gwots(gwots: pd.DataFrame, f0, f1, f2):
    w = wio.pstep_factory.get().wot
    gwots[w.p_resist] = vehicle.calc_p_resist(gwots.index, f0, f1, f2)
    return gwots


@autographed(provides="p_inert")
def calc_inertial_power(V, A, test_mass, f_inertial):
    """
    @see: Annex 2-3.1
    """
    return (A * V * test_mass * f_inertial) / 3600.0


@autographed(provides="p_req")
def calc_required_power(p_resist: Column, p_inert: Column):
    """
    Equals :math:`road_loads + inertial_power`

    @see: Annex 2-3.1
    """
    return p_resist + p_inert


def calc_default_resistance_coeffs(test_mass, regression_curves):
    """
    Approximating typical P_resistance based on vehicle test-mass.

    The regression-curves are in the model `resistance_coeffs_regression_curves`.
    Use it for rough results if you are missing the real vehicle data.
    """
    a = regression_curves
    f0 = a[0][0] * test_mass + a[0][1]
    f1 = a[1][0] * test_mass + a[1][1]
    f2 = a[2][0] * test_mass + a[2][1]

    return (f0, f1, f2)
