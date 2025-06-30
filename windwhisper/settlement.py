"""
Fetch layer representing human settlements.
"""

import rioxarray
import pyproj
from shapely.geometry import box
from shapely.ops import transform
import xarray as xr
from . import DATA_DIR

import xarray as xr


def get_population_subset(bbox: box):
    """
    Fast version for EPSG:4326 (lat/lon) NetCDF: slices lat/lon bounding box directly.

    Parameters:
        lat_min, lat_max, lon_min, lon_max (float): Bounding box in degrees

    Returns:
        xarray.DataArray: Subset of population data within the bounding box
    """

    lon_min, lat_min, lon_max, lat_max = bbox.bounds

    filepath = DATA_DIR / "GHS_POP_E2030_Europe.nc"

    # Open NetCDF lazily
    da = xr.open_dataset(filepath)["population"]

    # Make sure latitude is decreasing (common in raster data)
    lat0, lat1 = sorted([lat_min, lat_max], reverse=True)
    lon0, lon1 = sorted([lon_min, lon_max])

    # Slice using .sel() on lat/lon
    subset = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))

    return subset
