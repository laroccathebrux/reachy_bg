"""Make the GStreamer Python bindings importable in this venv on macOS.

The ``reachy-mini`` SDK gets GStreamer from pip wheels whose ``.pth`` file adds the bindings
(``gi``) to ``sys.path`` at start-up. On this Mac the ``.pth`` files of the venv end up with
the Finder ``hidden`` flag, and Python 3.12.13+ deliberately ignores hidden ``.pth`` files, so
``import gi`` fails and the robot daemon cannot start.

This script writes a ``sitecustomize.py`` into the venv's site-packages that performs the
same set-up the ``.pth`` would have done. ``sitecustomize`` is imported as a normal module,
so the hidden flag does not affect it. Idempotent; run it after every ``uv sync``:

    uv run python scripts/venv_postinstall.py
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

SITECUSTOMIZE = '''"""Written by scripts/venv_postinstall.py: replays the venv .pth files that macOS hides."""

import os
import sys

_SITE = os.path.dirname(os.path.abspath(__file__))

# 1. gstreamer wheels: put the GStreamer Python bindings (gi) on sys.path.
try:
    import gstreamer_libs

    gstreamer_libs.setup_python_environment()
except Exception:
    pass

# 2. Any other .pth file in this directory: execute its "import" lines and add its paths,
#    which is what site.py would do if the files were not hidden.
for _name in sorted(os.listdir(_SITE)):
    if not _name.endswith(".pth") or _name.startswith("gstreamer"):
        continue
    try:
        with open(os.path.join(_SITE, _name), encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                if _line.startswith(("import ", "import\\t")):
                    exec(_line)
                else:
                    _path = os.path.join(_SITE, _line)
                    if os.path.isdir(_path) and _path not in sys.path:
                        sys.path.append(_path)
    except Exception:
        pass
'''


def main() -> int:
    if sys.prefix == sys.base_prefix:
        print("not inside a virtual environment; run with `uv run python scripts/venv_postinstall.py`")
        return 1
    site_packages = Path(site.getsitepackages()[0])
    target = site_packages / "sitecustomize.py"
    if target.exists() and target.read_text(encoding="utf-8") == SITECUSTOMIZE:
        print(f"already in place: {target}")
    else:
        target.write_text(SITECUSTOMIZE, encoding="utf-8")
        print(f"written: {target}")
    hidden = [p.name for p in site_packages.glob("*.pth")]
    print(f".pth files replayed by sitecustomize: {', '.join(hidden) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
