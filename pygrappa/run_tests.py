import pathlib
import sys

import pytest


if __name__ == "__main__":
    this_dir = pathlib.Path(__file__).parent.resolve()
    retcode = pytest.main([f"{this_dir / 'tests'}"])
    sys.exit(retcode)
