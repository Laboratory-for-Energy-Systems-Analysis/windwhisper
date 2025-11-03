# -- Path setup --------------------------------------------------------------
from __future__ import annotations
import os
import sys
from pathlib import Path

# Only add the *project root* to sys.path (not the package dir)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------
# Keep this import light; don't import heavy submodules at package import time
try:
    from importlib.metadata import version, PackageNotFoundError  # py3.8+
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

project = "windwhisper"
author = "Windwhisper contributors"

try:
    release = version("windwhisper")
except PackageNotFoundError:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
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

# Mock heavy/optional runtime deps so autosummary can import your package
autodoc_mock_imports = [
    "numpy", "scipy", "pandas", "matplotlib",
    "shapely", "geopandas", "rasterio", "pyproj",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# Use the default theme (keeps requirements simple). If you prefer RTD theme:
# html_theme = "sphinx_rtd_theme"
