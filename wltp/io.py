#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
utilities for starting-up, parsing, naming, indexing and spitting out data

.. testsetup::

  from wltp.io import *
"""
import contextvars
import dataclasses
import itertools as itt
import re
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandalone import mappings

#: Contains all path/column names used, after code has run code.
#: Don't use it directly, but either
#: - through context-vars to allow for redefinitions, or
#: - call :func:`paths_collected()` at the end of a code run.
_root_pstep = mappings.Pstep()

#: The root-path wrapped in a context-var so that client code
#: can redfine paths & column names momentarily with::
#:
#:      token = wio.pstep_factory.set(mapping.Pstep(<mappings>))
#:      try:
#:          ...
#:      finally:
#:          wio.pstep_factory.reset(token)
#:         ...
pstep_factory = contextvars.ContextVar("root", default=_root_pstep)


def paths_collected(with_orig=False, tag=None) -> List[str]:
    """
    Return path/column names used, after code has run code.

    See :meth:`mappings.Pstep._paths`.
    """
    return _root_pstep._paths(with_orig, tag)


def getdval(mdl, key, default):
    """Returns `default` if `key` not in `mdl` or value is None/NAN"""
    if key in mdl:
        val = mdl[key]
        if val is not None and (not np.isscalar(val) or not np.isnan(val)):
            return val
    return default


def veh_name(v):
    n = pstep_factory.get().names
    v = int(v)
    return f"{n.v}{v:0>3d}"


def veh_names(vlist):
    return [veh_name(v) for v in vlist]


GearGenerator = Callable[[int], str]


def gear_name(g: int) -> str:
    n = pstep_factory.get().names
    return f"{n.g}{g}"


def gear_names(glist):
    return [gear_name(g) for g in glist]


def class_part_name(part_index):
    n = pstep_factory.get().names
    return f"{n.phase_}{part_index}"


def flatten_columns(columns, sep="/"):
    """Use :func:`inflate_columns()` to inverse it"""

    def join_column_names(name_or_tuple):
        if isinstance(name_or_tuple, tuple):
            return sep.join(n for n in name_or_tuple if n)
        return name_or_tuple

    return [join_column_names(names) for names in columns.to_flat_index()]


def inflate_columns(columns, levels=2, sep="/"):
    """Use :func:`flatten_columns()` to inverse it"""

    def split_column_name(name):
        assert isinstance(name, str), ("Inflating Multiindex?", columns)
        names = name.split(sep)
        if len(names) < levels:
            nlevels_missing = levels - len(names)
            names.extend([""] * nlevels_missing)
        return names

    tuples = [split_column_name(names) for names in columns]
    return pd.MultiIndex.from_tuples(tuples, names=["gear", "item"])


@dataclasses.dataclass(frozen=True, eq=True)
class GearMultiIndexer:
    """
    Multi-indexer for 2-level df columns like ``(item, gear)`` with 1-based & closed-bracket `gear`.

    Example *grid_wots*::

        p_avail  p_avail  ... n_foo  n_foo
             g1       g2  ...    g1     g2

    ... Warning::
        negative indices might not work as expected if :attr:`gnames` do not start from ``g1``
        (e.g. when constructed with :meth:`from_df()` static method)


    **Examples:**

    - Without `items` you get simple gear-indexing:

      >>> G = GearMultiIndexer.from_ngears(5)
      >>> G.gnames
      1    g1
      2    g2
      3    g3
      4    g4
      5    g5
      dtype: object
      >>> G[1:3]
      ['g1', 'g2', 'g3']
      >>> G[::-1]
      ['g5', 'g4', 'g3', 'g2', 'g1']
      >>> G[3:2:-1]
      ['g3', 'g2']
      >>> G[3:]
      ['g3', 'g4', 'g5']
      >>> G[3:-1]
      ['g3', 'g4', 'g5']
      >>> G[-1:-2:-1]
      ['g5', 'g4']

      >>> G[[1, 3, 2]]
      ['g1', 'g3', 'g2']

      >>> G[-1]
      'g5'


    - When `items` are given, you get a "product" MultiIndex:

      >>> G.with_item("foo", "bar")[1:3]
      MultiIndex([('foo', 'g1'),
                  ('foo', 'g2'),
                  ('foo', 'g3'),
                  ('bar', 'g1'),
                  ('bar', 'g2'),
                  ('bar', 'g3')],
                 )
      >>> G.with_item("foo")[2]
      MultiIndex([('foo', 'g2')],
                 )


      Use no `items` to reset them:

      >>> G.with_item('foo').with_item()[:]
      ['g1', 'g2', 'g3', 'g4', 'g5']


    - Notice that **G0** changes "negative" indices:

      >>> G[[-5, -6, -7]]
      ['g1', 'g5', 'g4']
      >>> G = GearMultiIndexer.from_ngears(5, gear0=True)
      >>> G[:]
      ['g0', 'g1', 'g2', 'g3', 'g4', 'g5']
      >>> G[[-5, -6, -7]]
      ['g1', 'g0', 'g5']
    """

    #: 1st level column(s)
    items: Optional[Iterable[str]]
    #: 2-level columns; use a generator like :func:`gear_names()` (default)
    #:
    #: to make a :class:`pd.Series` like::
    #:
    #:     {1: 'g1', 2: 'g2', ...}
    gnames: pd.Series
    #: Setting it to a gear not in :attr:`gnames`, indexing with negatives
    #: may not always work.
    top_gear: int
    #: a function returns the string representation of a gear, like ``1 --> 'g1'``
    generator: GearGenerator

    @classmethod
    def from_ngears(
        cls,
        ngears: int,
        items: Iterable[str] = None,
        generator: GearGenerator = gear_name,
        gear0=False,
    ):
        return GearMultiIndexer(
            items,
            pd.Series({i: generator(i) for i in range(int(not gear0), ngears + 1)}),
            ngears,
            generator,
        )

    @classmethod
    def from_gids(
        cls,
        gids: Iterable[int],
        items: Iterable[str] = None,
        generator: GearGenerator = gear_name,
    ):
        gids = sorted(gids)
        gids = pd.Series({i: generator(i) for i in gids})
        return GearMultiIndexer(items, gnames, gids[-1], generator)

    @classmethod
    def from_df(
        cls, df, items: Iterable[str] = None, generator: GearGenerator = gear_name
    ):
        """
        Derive gears from the 2nd-level columns, sorted, and the last one becomes `ngear`

        :param df:
            the 2-level df, not stored, just to get gear-names.

        ... Warning::
            Negative indices might not work as expected if :attr:`gnames`
            does not start from ``g1``.
        """
        gears = [g for g in df.columns.levels[1] if g]
        gids = [int(i) for i in re.sub("[^0-9 ]", "", " ".join(gears)).split()]
        gnames = pd.Series(gears, index=gids).sort_index()
        return cls(items, gnames, gids[-1], generator)

    def with_item(self, *items: str):
        return type(self)(items or None, self.gnames, self.top_gear, self.generator)  # type: ignore

    def __getitem__(self, key):
        """
        1-based & closed-bracket indexing, like Series but with `-1` for the top-gear.
        """
        top_gear = self.ng
        # Support partial gears or G0!
        offset = int(top_gear == self.top_gear)

        def from_top_gear(i):
            return offset + (i % top_gear) if isinstance(i, int) and i < 0 else i

        if isinstance(key, slice):
            key = slice(from_top_gear(key.start), from_top_gear(key.stop), key.step)
        elif isinstance(key, int):
            key = from_top_gear(key)
        else:  # assume Iterable[int]
            key = [from_top_gear(g) for g in key]

        gnames = self.gnames.loc[key]

        ## If no items, return just a list of gears.
        #
        if self.items is None:
            if isinstance(gnames, pd.Series):
                gnames = list(gnames)
            return gnames

        ## Otherwise, return a product multi-index.
        #
        if not isinstance(gnames, pd.Series):
            gnames = (gnames,)
        return pd.MultiIndex.from_tuples(itt.product(self.items, gnames))

    def colidx_pairs(
        self, item: Union[str, Iterable[str]], gnames: Iterable[str] = None
    ):
        if gnames is None:
            gnames = self.gnames
        assert gnames, locals()

        if isinstance(item, str):
            item = (item,)
        return pd.MultiIndex.from_tuples(itt.product(item, gnames))

    @property
    def ng(self):
        """
        The number of gears extracted from 2-level dataframe.

        It equals :attr:`top_gear` if :attr:`gnames` are from 1-->top_gear.
        """
        return len(self.gnames)

def make_autograph(*args, **kw):
    """Configures a new :class:`.Autograph` with func-name patterns for this project. """
    from .autograph import Autograph

    return Autograph(
        [
            "get_",
            "calc_",
            "upd_",
            "create_",
            "decide_",
            re.compile(r"\battach_(\w+)_in_(\w+)$"),
        ], *args, **kw
    )