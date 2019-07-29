#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import io
import os
import sys
from typing import Union


##############
#  Utilities
#
def str2bool(v):
    import argparse

    vv = v.lower()
    if vv in ("yes", "true", "on"):
        return True
    if vv in ("no", "false", "off"):
        return False
    try:
        return float(v)
    except:
        raise argparse.ArgumentTypeError("Invalid boolean(%s)!" % v)


def pairwise(t):
    """From http://stackoverflow.com/questions/4628290/pairs-from-single-list"""
    it1 = iter(t)
    it2 = iter(t)
    try:
        next(it2)
    except:
        return []
    return zip(it1, it2)


## From http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
#
def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """

    class memodict(dict):
        def __init__(self, f):
            self.f = f

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret

    return memodict(f)


## From http://stackoverflow.com/a/4149190/548792
#
class Lazy(object):
    def __init__(self, func):
        self.func = func

    def __str__(self):
        return self.func()


def is_travis():
    return "TRAVIS" in os.environ


def generate_filenames(filename):
    f, e = os.path.splitext(filename)
    yield filename
    i = 1
    while True:
        yield "%s%i%s" % (f, i, e)
        i += 1


def open_file_with_os(fpath):
    ## From http://stackoverflow.com/questions/434597/open-document-with-default-application-in-python
    #     and http://www.dwheeler.com/essays/open-files-urls.html
    import subprocess

    try:
        os.startfile(fpath)
    except AttributeError:
        if sys.platform.startswith("darwin"):
            subprocess.call(("open", fpath))
        elif os.name == "posix":
            subprocess.call(("xdg-open", fpath))
    return


def yaml_dump(o, fp):
    from ruamel import yaml
    from ruamel.yaml import YAML

    y = YAML(typ="rt", pure=True)
    y.default_flow_style = False
    # yaml.canonical = True
    yaml.scalarstring.walk_tree(o)
    y.dump(o, fp)


def yaml_dumps(o) -> str:
    fp = io.StringIO()
    yaml_dump(o, fp)
    return fp.getvalue()


def yaml_load(fp) -> Union[dict, list]:
    from ruamel.yaml import YAML

    y = YAML(typ="rt", pure=True)
    return y.load(fp)


def yaml_loads(txt: str) -> Union[dict, list]:
    fp = io.StringIO(txt)
    return yaml_load(fp)


def make_xy_df(data, xname=None, yname=None, auto_transpose=False):
    """
    Make a X-indexed df from 2D-matrix(lists/numpy), dict, df(1-or-2 cols) or series.

    :param auto_transpose:
        If not empty, ensure longer dimension is the rows (axis-0).
    """
    import pandas as pd

    def invalid_input_ex(df):
        cols_msg = ", ".join(
            f"{argname}={val!r}"
            for argname, val in [("xname", xname), ("yname", yname)]
            if val is not None
        )
        if cols_msg:
            cols_msg = f" with {cols_msg}"
        return ValueError(
            f"Expected a df/series{cols_msg} (got type: {type(data)}), or 2 columns at most (got {ncols}: {df.columns})!"
        )

    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        if auto_transpose and not df.empty and df.shape[0] < ncols:
            df = df.T

        has_index = isinstance(data, (pd.Series, pd.DataFrame))
        ncols = df.shape[1]
        if ncols == 0:
            #
            ## Handle empties
            if yname is None:
                yname = 0  # default name for the 1st column.
            df[yname] = pd.np.NaN
        elif xname == yname == None:
            if ncols > 2:
                raise invalid_input_ex(df)
            elif ncols == 1:
                if not has_index:
                    # Accept index as X only if given by user.
                    raise invalid_input_ex(df)
        elif xname is None and yname is not None:
            if yname in df.columns:
                y = df.loc[:, yname].to_frame()
                if ncols == 2:
                    y.index = df.drop(yname, axis=1).squeeze()
                    df = y
                elif has_index:
                    df = y
                else:
                    raise invalid_input_ex(df)
            elif ncols > 2 or not (ncols == 1 and has_index):
                raise invalid_input_ex(df)
        elif xname is not None and yname is None:
            if xname in df.columns:
                if ncols == 2:
                    x = df.loc[:, xname]
                    df = df.drop(xname, axis=1)
                    df.index = x
                else:
                    raise invalid_input_ex(df)
            elif ncols != 2:
                raise invalid_input_ex(df)
        elif yname in df.columns and xname in df.columns:
            df = df.loc[:, [xname, yname]]
            df.set_index(xname)
        elif yname in df.columns and xname == df.index.name:
            df = df.loc[:, yname].to_frame()
        elif yname in df.columns:
            y = df.loc[:, yname].to_frame()
            if ncols == 2:
                y.index = df.drop(yname, axis=1).squeeze()
                df = y
            elif has_index:
                df = y
            else:
                raise invalid_input_ex(df)
        elif xname in df.columns:
            if ncols == 2:
                x = df.loc[:, xname]
                df = df.drop(xname, axis=1)
                df.index = x
            else:
                raise invalid_input_ex(df)
        elif ncols != 2 or not (ncols == 1 and has_index):
            raise invalid_input_ex(df)

        if df.shape[1] == 2:
            df = df.set_index(df.columns[0])

        if yname is not None:
            df.columns = [yname]

        if xname is not None:
            df.index.name = xname

        return df
    except BrokenPipeError as ex:
        if ex.args and ex.args[0].startswith("Expected a df/series"):
            raise
        raise ValueError(f"Invalid XY input(type: {type(data)}), due to: {ex}") from ex
