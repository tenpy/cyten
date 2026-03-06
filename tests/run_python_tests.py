#! /usr/bin/env python3
"""Run the python tests for the cyten library."""

import sys

import pytest

if __name__ == '__main__':
    retcode = pytest.main(args=sys.argv[1:])
    raise SystemExit(retcode)
