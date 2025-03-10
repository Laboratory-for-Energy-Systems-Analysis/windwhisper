"""
Estimate noise attenuation due to ground type, distance from the source and elevation difference between the source
and receiver, according to ISO 9613-2:2024.
"""


from multiprocessing import Pool
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from windwhisper.elevation_grid import get_elevation_grid, distances_with_elevation

# Move compute_turbine_attenuation outside
def compute_turbine_attenuation(args):
    turbine, specs, latitudes, longitudes, elevation_grid, interpolator, euclidian_distances = args
    source_lat, source_lon = specs["position"]
    source_elevation = interpolator([source_lat, source_lon]) + specs["hub height"]

    ground_attenuation = np.zeros((len(latitudes), len(longitudes)))
    obstacles_attenuation = np.zeros((len(latitudes), len(longitudes)))

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            # Skip source point
            if (lon, lat) == (source_lon, source_lat):
                continue

            # Receiver elevation
            receiver_elevation = elevation_grid.values[i, j]
            path_latitudes = np.linspace(source_lat, lat, 100)
            path_longitudes = np.linspace(source_lon, lon, 100)
            path_coords = np.column_stack((path_latitudes, path_longitudes))

            # Elevation profile along the path
            path_elevations = interpolator(path_coords)
            straight_elevation = np.squeeze(np.linspace(source_elevation, receiver_elevation, path_elevations.size))

            # Obstacle detection
            obstacle_mask = path_elevations > (straight_elevation + 5)  # Boolean mask of obstacles

            if np.any(obstacle_mask):  # Check if there are any obstacles
                #print(obstacle_mask.sum())
                # Calculate distances for diffraction
                obstacle_index = np.argmax(obstacle_mask)
                d_ss = obstacle_index / len(path_elevations) * euclidian_distances.values[i, j]
                d_sr = euclidian_distances.values[i, j] - d_ss
                d = euclidian_distances.values[i, j]

                # Calculate path length difference
                z = d_ss + d_sr - d

                # Use ISO 9613-2 barrier attenuation formula for dBA
                lambda_eff = 0.34  # Effective wavelength for A-weighted spectrum (1 kHz)
                a_bar_dba = 10 * np.log10(3 + 40 * (z / lambda_eff))

                obstacles_attenuation[i, j] = max(0, a_bar_dba)

            # Area between path and straight line
            area = np.clip(np.trapz(straight_elevation - path_elevations, dx=1), 0, None)
            mean_height = area / euclidian_distances.values[i, j]
            a_gr = 4.8 - ((2 * mean_height) / euclidian_distances.values[i, j]) * (17 + (300 / euclidian_distances.values[i, j]))
            ground_attenuation[i, j] = max(0, a_gr)

    return ground_attenuation, obstacles_attenuation


def calculate_ground_attenuation(
        haversine_distances,
        longitudes,
        latitudes,
        wind_turbines,
        elevation_data
):
    """
    Calculate the ground attenuation in dB, according to ISO 9613-2:2024.
    """

    elevation_grid = get_elevation_grid(
        longitudes=longitudes,
        latitudes=latitudes,
        elevation_data=elevation_data
    )

    if elevation_grid is None:
        return None, None, None

    # Precompute interpolator
    interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        elevation_grid.values
    )

    # Precompute relative elevations
    relative_elevations = elevation_grid - xr.concat(
        [
            elevation_grid.interp(
                coords={"lat": specs["position"][0], "lon": specs["position"][1]}
            ).expand_dims(dim={"turbine": [turbine_name]})
            for turbine_name, specs in wind_turbines.items()
        ],
        dim="turbine",
    )

    euclidian_distances = distances_with_elevation(haversine_distances, relative_elevations)

    # Prepare arguments for parallel execution
    args = [
        (
            turbine,
            specs,
            latitudes,
            longitudes,
            elevation_grid,
            interpolator,
            euclidian_distances.sel(turbine=turbine)
        )
        for turbine, specs in wind_turbines.items()
    ]

    # Parallelize turbine computation
    with Pool() as pool:
        results = pool.map(compute_turbine_attenuation, args)

    # Aggregate results
    ground_attenuation = np.min([res[0] for res in results], axis=0)
    obstacles_attenuation = np.min([res[1] for res in results], axis=0)

    return elevation_grid, xr.DataArray(ground_attenuation, dims=("lat", "lon"), coords={"lat": latitudes, "lon": longitudes}), xr.DataArray(obstacles_attenuation, dims=("lat", "lon"), coords={"lat": latitudes, "lon": longitudes})
