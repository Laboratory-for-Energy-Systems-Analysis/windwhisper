"""Sphinx configuration for the WindWhisper documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata

# Ensure the project root is on the path so autodoc can import the package.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

project = "WindWhisper"

try:
    release = metadata.version("windwhisper")
except metadata.PackageNotFoundError:  # pragma: no cover - during local builds without installation
    release = "0.0.0"

copyright = f"{datetime.now():%Y}, Paul Scherrer Institute"
author = "Laboratory for Energy Systems Analysis"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
    "pandas": ("https://pandas.pydata.org/docs", {}),
    "xarray": ("https://docs.xarray.dev/en/stable", {}),
}

# Mock heavy scientific dependencies so the documentation can build without
# requiring compiled binaries in the Read the Docs environment.
autodoc_mock_imports = [
    "folium",
    "geopandas",
    "matplotlib",
    "pandas",
    "netCDF4",
    "osmnx",
    "pyproj",
    "rasterio",
    "scikit_learn",
    "seaborn",
    "sklearn",
    "skops",
    "xarray",
]

# Delegate type hint formatting to the descriptions for better readability.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Allow Markdown files via MyST.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = True

