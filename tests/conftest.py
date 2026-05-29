"""pytest configuration for viz_authoring tests."""
import sys
from pathlib import Path

# Make viz_authoring importable when running pytest from the repo root
# (where flame_sheep's pyproject defines viz_authoring/tests as a
# possible target). The package is installed via `pip install -e
# ./viz_authoring` for normal use; this is just a defensive add.
_src = Path(__file__).resolve().parent.parent / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
