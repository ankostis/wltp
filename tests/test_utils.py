#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import numpy as np
import pandas as pd
import pytest

from wltp import utils

_x = [0.1, 0.11]
_y = [0.2, 0.21]
_xy = np.array([_x, _y]).T


def _check_labels(df, xname, yname):
    if xname is not None:
        assert df.index.name == xname
    if yname is not None:
        assert df.columns == [yname]


@pytest.fixture(params=[None, "a"])
def xname(request):
    return request.param


@pytest.fixture(params=[None, "b"])
def yname(request):
    return request.param


@pytest.mark.parametrize(
    "xy",
    [
        _xy,
        _xy.tolist(),
        pd.DataFrame(_xy),
        pd.DataFrame(_y, index=_x),
        pd.Series(dict(_xy.tolist())),
    ],
)
def test_make_xy_df_simple(xy, xname, yname):
    res = utils.make_xy_df(xy, xname, yname)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (2, 1)
    assert res.values.T.tolist() == [_y]
    assert res.index.tolist() == _x
    _check_labels(res, xname, yname)


def _DF(*args, idxname=None, **kwargs):
    df = pd.DataFrame(*args, **kwargs)
    df.index.name = idxname
    return df


@pytest.mark.parametrize(
    "make_xy_args, exp",
    [
        (([1],), ValueError),
        (pd.DataFrame(([1])), pd.DataFrame([1])),
        (([1, 2, 3],), ValueError),
        (pd.Series(([1, 2, 3])), pd.DataFrame([[1], [2], [3]])),
        (([1], None, "ab"), ValueError),
        ((pd.Series([1]), None, "ab"), pd.DataFrame([1], columns=["ab"])),
        (([1], "X", None), ValueError),
        ((pd.DataFrame([1]), "X", None), _DF([1], columns=[[0]], idxname="X")),
        (({"a": [1, 2]},), pd.DataFrame([[1], [2]], columns=["a"])),
        (({"a": [1, 2]}, "X", "Y"), _DF([[1], [2]], columns=["Y"], idxname="X")),
        (([[11, 12], [21, 22]],), pd.DataFrame([[11, 12], [21, 22]]).set_index(0)),
        (
            (pd.DataFrame({"YY": [11, 22], "extra": [0, 0], "XX": [3, 4]}), "XX", "YY"),
            _DF({"YY": [11, 22]}, index=[3, 4], idxname="XX"),
        ),
        (
            (
                _DF({"Y": [1, 2], "extra": [5, 6]}, index=[11, 22], idxname="XX"),
                "XX",
                "Y",
            ),
            _DF({"Y": [1, 2]}, index=[11, 22], idxname="XX"),
        ),
        ((pd.DataFrame({"YY": [11, 22], "extra": [0, 0], "XX": [3, 4]}),), ValueError),
    ],
)
def test_make_xy_df(make_xy_args, exp):
    if type(exp) is type and issubclass(exp, Exception):
        with pytest.raises(exp):
            utils.make_xy_df(*make_xy_args)
    else:
        res = utils.make_xy_df(*make_xy_args)
        assert isinstance(res, pd.DataFrame)
        assert exp.equals(res)
        assert res.columns == exp.columns
        assert (res.index == exp.index).all()
        if exp.index.name is not None:
            assert res.index.name == exp.index.name


@pytest.mark.parametrize(
    "xy, shape", [([], (0, 1)), ([[]], (1, 1)), ([[], []], (2, 1))]
)
def test_make_xy_df_empties(xy, xname, yname, shape):
    res = utils.make_xy_df(xy, xname, yname)
    assert not res.empty or xy == []
    assert res.shape == shape
    _check_labels(res, xname, yname)


@pytest.mark.parametrize("xy", [1, [[1, 2, 3], [3, 4, 5], [5, 6, 7]]])
def test_make_xy_df_errors(xy, xname, yname):
    with pytest.raises(ValueError):
        utils.make_xy_df(xy, xname, yname)
