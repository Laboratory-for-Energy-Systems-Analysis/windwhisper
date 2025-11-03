from __future__ import annotations
import sys
from pathlib import Path

# ---- Path setup: add PROJECT ROOT only (not the package dir) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Project info ----
project = "windwhisper"
author = "Windwhisper contributors"

# Avoid naming collision with importlib.metadata.version
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    release = _pkg_version("windwhisper")
except Exception:
    # Fallback during RTD cold builds if package isn't yet importable
    release = "0.0.0"

# ---- General config ----
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Mock heavy/optional deps to let imports succeed during docs build
autodoc_mock_imports = [
    "numpy", "scipy", "pandas", "matplotlib",
    "shapely", "geopandas", "rasterio", "pyproj",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# html_theme = "sphinx_rtd_theme"  # uncomment if you want RTD theme
