import logging
from pathlib import Path
from typing import Sequence as Seq

import pandas as pd

from . import vehdb

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def test_taskforce_vehs(
    h5_accdb: str,
    h5_pyalgo: str,
    h5_write: bool,
    vehnums_to_run: Seq[int],
    props_group_suffix="oprop",
    cycle_group_suffix="cycle",
    wots_vmax_group_suffix="wots_vmax",
):
    """
    RUN PYTHON on cars that have AccDB results in /vehicles/v123/out1

    and build:

        vehicles/
            +--v001/
            |   +--oprop      ADD: (series) scalars generated by python-algo
            |   +--cycle      ADD: (df) cycle-run generated by python-alog
            +...

    """

    def run_vehicles_with_pythons(h5db, vehnums):
        for vehnum in vehnums:
            log.info("Calculating veh(v%0.3i)...", vehnum)
            try:
                yield vehnum, vehdb.run_pyalgo_on_accdb_vehicle(inph5, vehnum)
            except Exception as ex:
                log.error("V%0.3i failed: %s", vehnum, ex)
                raise ex

    def compare_vehicle(h5db, vehnum, oprops, cycle, wots_vmax):
        oprops = pd.Series(oprops)
        db_oprops = h5db.get(vehdb.vehnode(vehnum, props_group_suffix))
        db_cycle = h5db.get(vehdb.vehnode(vehnum, cycle_group_suffix))
        db_wots_vmax = h5db.get(vehdb.vehnode(vehnum, wots_vmax_group_suffix))

        to_compare = [(oprops, db_oprops), (cycle, db_cycle), (wots_vmax, db_wots_vmax)]

        for calced, stored in to_compare:
            if not calced.equals(stored):
                df = pd.concat((calced, stored)).drop_duplicates(keep=False)
                raise Exception(f"Calced-vs-stored differ in {len(df)} rows!")

    def store_vehicle(h5db, vehnum, oprops, cycle, wots_vmax):
        log.info("STORING veh(v%0.3i)...", vehnum)

        g = vehdb.vehnode(vehnum, props_group_suffix)
        h5db.put(g, pd.Series(oprops))
        vehdb.provenir_h5node(h5db, g, title="Pyalgo generated")

        g = vehdb.vehnode(vehnum, cycle_group_suffix)
        h5db.put(g, cycle)
        vehdb.provenir_h5node(h5db, g, title="Pyalgo generated")

        g = vehdb.vehnode(vehnum, wots_vmax_group_suffix)
        h5db.put(g, wots_vmax)
        vehdb.provenir_h5node(h5db, g, title="Pyalgo generated")

    def store_or_compare_vehicle(*args, **kw):
        compared = False
        # File-check not really needed, would still raise KeyError.
        if not h5_write and Path(h5_pyalgo).exists():
            try:
                compare_vehicle(*args, **kw)
                compared = True
            except KeyError as ex:
                if "No object named" not in str(ex):
                    raise  # Scream, unless HDF5 file is missing nodes.

        if not compared:
            store_vehicle(*args, **kw)

    with vehdb.openh5(h5_accdb) as inph5, vehdb.openh5(h5_pyalgo) as outh5:
        vehnums = vehdb.all_vehnums(inph5) if vehnums_to_run is None else vehnums_to_run
        for vehnum, pyalgo_outs in vehdb.do_h5(
            inph5, run_vehicles_with_pythons, vehnums
        ):
            vehdb.do_h5(outh5, store_or_compare_vehicle, vehnum, *pyalgo_outs)
