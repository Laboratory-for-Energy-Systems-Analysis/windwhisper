"""Utilities for retrieving and interpolating elevation grids."""

import requests
import numpy as np
import xarray as xr
import os
from dotenv import load_dotenv
from xarray import DataArray
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pyproj import Geod


from .utils import load_secret

load_dotenv()

GOOGLE_API_KEY = load_secret()

if GOOGLE_API_KEY:
    ELEVATION_API = os.getenv("API_ELEVATION_GOOGLE")
    location_separator, location_extra = "%2C", "%7C"
    #print("Using Google Elevation API")
else:
    ELEVATION_API = os.getenv("API_ELEVATION")
    location_separator, location_extra = ",", "|"
    #print("Using Open Elevation API")

MAX_SAMPLING_POINTS = int(os.getenv("MAX_SAMPLING_POINTS"))


def get_elevation_grid(
        longitudes: np.array,
        latitudes: np.array,
        elevation_data = None,
        wind_turbines: dict = None
) -> DataArray | None:
    """Return a terrain elevation grid for the target domain.

    :param longitudes: Longitudes defining the interpolation grid.
    :type longitudes: numpy.ndarray
    :param latitudes: Latitudes defining the interpolation grid.
    :type latitudes: numpy.ndarray
    :param elevation_data: Optional local dataset used instead of remote APIs.
    :type elevation_data: xarray.DataArray | None
    :param wind_turbines: Turbine specifications including ``position`` tuples.
    :type wind_turbines: dict | None
    :returns: Elevation grid interpolated to the provided coordinates or
        ``None`` when the remote request fails.
    :rtype: xarray.DataArray | None
    """

    # determine the bounding box
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    longitudes_ = np.linspace(min_lon, max_lon, MAX_SAMPLING_POINTS)
    latitudes_ = np.linspace(min_lat, max_lat, MAX_SAMPLING_POINTS)

    # create a grid of latitudes and longitudes
    path = [(lat, lon) for lon in longitudes_ for lat in latitudes_]

    if elevation_data is None:
        print("Using elevation API")
        # get elevation data from Open Elevation API
        url = ELEVATION_API
        for lat, lon in path:
            url += f"{lat}{location_separator}{lon}{location_extra}"  # append the coordinates to the URL
        url = url[:-len(location_extra)]  # remove the trailing separator

        if GOOGLE_API_KEY:
            url += f"&key={GOOGLE_API_KEY}"

        response = requests.get(url)

        if response.status_code == 200:
            elevations = response.json()["results"]

            # Extract data into a DataFrame
            df = pd.DataFrame([
                {'lat': d['location']['lat'], 'lon': d['location']['lng'], 'elevation': d['elevation']}
                for d in elevations
            ])

            # Create a pivot table to reshape the data into a grid
            grid = df.pivot(index='lat', columns='lon', values='elevation')

            # Create the xarray DataArray
            da = xr.DataArray(
                data=grid.values,
                dims=["latitude", "longitude"],
                coords={
                    "latitude": grid.index.values,
                    "longitude": grid.columns.values,
                },
                name="elevation",
            )

            # interpolate to latitudes and longitudes
            da = da.interp(latitude=latitudes, longitude=longitudes)

            return da

        else:
            print(f"Failed to fetch elevation data: {response.status_code}")
            return None

    else:
        # we use directly elevation data
        print("Using local elevation data")

        elevation_data = clip_array_around_turbines(
            elevation_data,
            wind_turbines
        )

        # Ensure ascending order
        if elevation_data.latitude.values[0] > elevation_data.latitude.values[-1]:
            elevation_data = elevation_data.sortby("latitude")
        if elevation_data.longitude.values[0] > elevation_data.longitude.values[-1]:
            elevation_data = elevation_data.sortby("longitude")

        grid_data = np.asarray(elevation_data.data)

        interpolator = RegularGridInterpolator(
            (elevation_data.latitude.values, elevation_data.longitude.values),
            grid_data,
            bounds_error=False,
            fill_value=np.nan
        )

        # Build lat/lon mesh grid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Flatten and stack coordinates for interpolation
        points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        interpolated_values = interpolator(points).reshape(lat_grid.shape)

        # Return as xarray DataArray
        da_interp = xr.DataArray(
            interpolated_values,
            coords={"latitude": latitudes, "longitude": longitudes},
            dims=["latitude", "longitude"],
            name="elevation"
        )

        return da_interp


def clip_array_around_turbines(da, wind_turbines, buffer_km=5.0):
    """Clip an elevation grid around the wind turbines.

    :param da: Elevation array with ``latitude`` and ``longitude`` coordinates.
    :type da: xarray.DataArray | xarray.Dataset
    :param wind_turbines: Turbine specifications including ``position`` tuples
        expressed as ``(lat, lon)``.
    :type wind_turbines: dict
    :param buffer_km: Distance around each turbine to retain, in kilometres.
    :type buffer_km: float
    :returns: Elevation subset covering all turbines and their buffers.
    :rtype: xarray.DataArray
    :raises ValueError: If the dataset does not contain an ``'elevation'``
        variable when a dataset is provided instead of a data array.
    """
    geod = Geod(ellps="WGS84")

    min_lat, max_lat = 90, -90
    min_lon, max_lon = 180, -180

    for turbine in wind_turbines.values():
        lat, lon = turbine["position"]

        # Approximate buffer bounding box using geodesic projection
        lats = []
        lons = []
        for azimuth in [0, 90, 180, 270]:
            lon_off, lat_off, _ = geod.fwd(lon, lat, azimuth, buffer_km * 1000)
            lats.append(lat_off)
            lons.append(lon_off)

        min_lat = min(min_lat, min(lats))
        max_lat = max(max_lat, max(lats))
        min_lon = min(min_lon, min(lons))
        max_lon = max(max_lon, max(lons))

    # Prepare interpolator (scipy needs values as (lat, lon))
    if isinstance(da, xr.Dataset):
        try:
            da = da["elevation"]
        except KeyError:
            raise ValueError("DataArray must contain 'elevation' variable."
                             "Current variables: " + str(da.data_vars))

    # Ensure latitudes are sorted in ascending order before slicing
    if da.latitude[0] > da.latitude[-1]:
        da = da.sortby("latitude")

    clipped = da.sel(
        latitude=slice(min_lat, max_lat),
        longitude=slice(min_lon, max_lon)
    )

    return clipped


def distances_with_elevation(distances, relative_elevations):
    """Compute 3D distances using surface distance and elevation deltas.

    :param distances: Surface distances between points (typically from
        haversine calculations).
    :type distances: xarray.DataArray
    :param relative_elevations: Elevation differences between the same point
        pairs.
    :type relative_elevations: xarray.DataArray
    :returns: Distances adjusted for the elevation component.
    :rtype: xarray.DataArray
    :raises AssertionError: If the provided arrays do not share the same shape.
    """
    # Ensure both arrays have the same shape
    assert distances.shape == relative_elevations.shape, "Shapes must match!"

    # Perform element-wise operation without expanding dimensions
    euclidean_distances = np.sqrt(distances.data ** 2 + relative_elevations.data ** 2)

    # Return as an xarray DataArray to maintain metadata
    return xr.DataArray(euclidean_distances, coords=distances.coords, dims=distances.dims)


