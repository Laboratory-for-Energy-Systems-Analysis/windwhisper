from __future__ import annotations
import os
import sys
from pathlib import Path

# ---- Path setup: add PROJECT ROOT only (not the package dir) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Flag so your package (optionally) skips heavy init during docs builds
os.environ.setdefault("WINDWHISPER_DOCS", "1")

# ---- Project info ----
project = "windwhisper"
author = "Windwhisper contributors"

# Robust version retrieval; harmless if the dist isn't importable yet
try:
    from importlib.metadata import version as _pkg_version  # py3.8+
    release = _pkg_version("windwhisper")
except Exception:
    release = "0.0.0"

# ---- General config ----
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# IMPORTANT: Do NOT mock numpy/pandas/matplotlib here.
# Mock only heavy/optional libs that cause binary import issues on RTD.
autodoc_mock_imports = [
    "geopandas",
    "rasterio",
    "pyproj",
    "shapely",
    "osmnx",
    "netCDF4",
    "xarray",
    "sklearn",
    "folium",
    "ipywidgets",
    "pyogrio",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme (optional)
html_theme = "sphinx_rtd_theme"
