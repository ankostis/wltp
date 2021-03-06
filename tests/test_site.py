#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2020 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import io
import os.path as osp
import re
import subprocess
import sys

import pytest
from readme_renderer import rst

import wltp
from wltp import cli

mydir = osp.dirname(__file__)
proj_path = osp.join(mydir, "..")
readme_path = osp.join(proj_path, "README.rst")


def _badge_io_escape(s: str) -> str:
    return s.replace("-", "--").replace("_", "__").replace(" ", "_")


def test_README_proj_version_reldate():
    ver = _badge_io_escape(wltp.__version__)
    reldate = _badge_io_escape(wltp.__updated__)
    tail_len = 100
    mydir = osp.dirname(__file__)
    ver_found = rdate_found = False
    with open(readme_path) as fd:
        lines = fd.readlines()
    tail = lines[-tail_len:]
    for i, l in enumerate(tail, len(lines) - tail_len):
        if ver in l:
            ver_found = True
        if reldate not in l:
            rdate_found = True

    if not ver_found:
        msg = "Version(%s) not found in README %s tail-lines!"
        raise AssertionError(msg % (ver, tail_len))
    if not rdate_found:
        msg = "RelDate(%s) not found in README %s tail-lines!"
        raise AssertionError(msg % (reldate, tail_len))


def test_README_version_from_cmdline(capsys):
    ver = wltp.__version__
    with open(readme_path) as fd:
        text = fd.read()
        try:
            cli.main(["--version"])
        except SystemExit:
            pass  ## Cancel argparse's exit()
        captured = capsys.readouterr()
        proj_ver = captured.out.strip()
        assert proj_ver
        assert proj_ver in text, (
            "Version(%s) not found in README cmd-line version-check!" % ver,
        )


########################
## Copied from Twine

# Regular expression used to capture and reformat docutils warnings into
# something that a human can understand. This is loosely borrowed from
# Sphinx: https://github.com/sphinx-doc/sphinx/blob
# /c35eb6fade7a3b4a6de4183d1dd4196f04a5edaf/sphinx/util/docutils.py#L199
_REPORT_RE = re.compile(
    r"^<string>:(?P<line>(?:\d+)?): "
    r"\((?P<level>DEBUG|INFO|WARNING|ERROR|SEVERE)/(\d+)?\) "
    r"(?P<message>.*)",
    re.DOTALL | re.MULTILINE,
)


class _WarningStream:
    def __init__(self):
        self.output = io.StringIO()

    def write(self, text):
        matched = _REPORT_RE.search(text)

        if not matched:
            self.output.write(text)
            return

        self.output.write(
            "line {line}: {level_text}: {message}\n".format(
                level_text=matched.group("level").capitalize(),
                line=matched.group("line"),
                message=matched.group("message").rstrip("\r\n"),
            )
        )

    def __repr__(self):
        return self.output.getvalue()


@pytest.mark.skipif(sys.version_info < (3, 8), reason="assume build on py3.8 only")
def test_README_as_PyPi_landing_page(monkeypatch):
    """Not executing `setup.py build-sphinx` to control log/stderr visibility with pytest"""
    long_desc = subprocess.check_output(
        "python setup.py --long-description".split(), cwd=proj_path
    )
    assert long_desc is not None, "Long_desc is null!"

    err_stream = _WarningStream()
    result = rst.render(
        long_desc,
        # The specific options are a selective copy of:
        # https://github.com/pypa/readme_renderer/blob/master/readme_renderer/rst.py
        stream=err_stream,
        halt_level=2,  # 2=WARN, 1=INFO
    )
    assert result, err_stream


@pytest.mark.skipif(sys.version_info < (3, 8), reason="assume build on py3.8 only")
def test_sphinx_html():
    # Fail on warnings, but don't rebuild all files (no `-a`),
    subprocess.check_output("python setup.py build_sphinx -W".split(), cwd=proj_path)
