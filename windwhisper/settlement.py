"""
Fetch layer representing human settlements.
"""

from shapely.geometry import box
from . import DATA_DIR

import xarray as xr


def get_population_subset(bbox: box):
    """Extract population data within the requested bounding box.

    :param bbox: Bounding box in geographic coordinates (EPSG:4326).
    :type bbox: shapely.geometry.Polygon
    :returns: Population raster subset covering the bounding box.
    :rtype: xarray.DataArray
    """

    lon_min, lat_min, lon_max, lat_max = bbox.bounds

    filepath = DATA_DIR / "worldpop.nc"

    # Open NetCDF lazily
    da = xr.open_dataset(filepath)["population"]

    # Make sure latitude is decreasing (common in raster data)
    lat0, lat1 = sorted([lat_min, lat_max], reverse=True)
    lon0, lon1 = sorted([lon_min, lon_max])

    # Slice using .sel() on lat/lon
    subset = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))

    return subset
