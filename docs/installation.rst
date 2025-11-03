Installation
============

WindWhisper targets Python 3.10 or newer. The recommended way to install the
library and its dependencies is via ``pip`` within a virtual environment:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   pip install windwhisper

When working from a clone of the repository you can install the package in
editable mode together with the development extras used in the test suite:

.. code-block:: console

   pip install -e .[dev]

The documentation build on Read the Docs mocks heavy scientific dependencies
such as ``rasterio`` or ``geopandas``. When running the documentation locally you
may prefer to install those libraries explicitly so the API reference can access
their public attributes.
