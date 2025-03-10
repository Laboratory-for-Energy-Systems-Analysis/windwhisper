"""
Fetch elevation grid for an area.
"""

import requests
import numpy as np
import xarray as xr
import os
from dotenv import load_dotenv
from xarray import DataArray
import pandas as pd

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
        elevation_data = None
) -> DataArray | None:
    """
    Fetch elevation data for a given bounding box.

    :param: longitudes: A list of longitudes.
    :param: latitude: A list of latitudes.
    :return: A 2D numpy array containing the elevation data.
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
        new_coords = {"latitude": ("latitude", latitudes), "longitude": ("longitude", longitudes)}

        # Perform bilinear interpolation
        interpolated_ds = elevation_data.interp(new_coords, method="linear")
        return interpolated_ds.to_array().squeeze()


import numpy as np
import xarray as xr


def distances_with_elevation(distances, relative_elevations):
    """
    Based on elevation and haversine distance, calculate the distance between two points
    considering the elevation difference.
    """
    # Ensure both arrays have the same shape
    assert distances.shape == relative_elevations.shape, "Shapes must match!"

    # Perform element-wise operation without expanding dimensions
    euclidean_distances = np.sqrt(distances.data ** 2 + relative_elevations.data ** 2)

    # Return as an xarray DataArray to maintain metadata
    return xr.DataArray(euclidean_distances, coords=distances.coords, dims=distances.dims)


