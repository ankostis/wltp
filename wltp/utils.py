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

    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        if auto_transpose and not df.empty and df.shape[0] < df.shape[1]:
            df = df.T

        ## Handle empties
        #
        if df.shape[1] == 0:
            if yname is None:
                yname = 0  # default name for the 1st column
            df[yname] = pd.np.NaN
        else:
            if df.shape[1] > 2:
                if not xname == yname == None:
                    if xname not in df.columns or yname not in df.columns:
                        raise ValueError(
                            f"Columns X={xname}, Y={yname} not found in {df.columns}"
                        )
                    else:
                        df = df.loc[:, [xname, yname]]
                        df.set_index(xname)
                else:
                    raise ValueError(
                        f"Expected 2 columns at most, not {df.shape[1]}: {df.columns}"
                    )
            if df.shape[1] == 2:
                df = df.set_index(df.columns[0])

            if yname is not None:
                df.columns = [yname]

        if xname is not None:
            df.index.name = xname

        return df
    except Exception as ex:
        raise ValueError(f"Invalid XY input(type: {type(data)}), due to: {ex}") from ex
