import numpy as np
from haversine import haversine, Unit
import ipywidgets as widgets
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
from dotenv import load_dotenv
import os

from .atmospheric_absorption import get_absorption_coefficient
from .geometric_divergence import get_geometric_spread_loss
from .ground_attenuation import calculate_ground_attenuation

load_dotenv()

NOISE_MAP_RESOLUTION = int(os.getenv("NOISE_MAP_RESOLUTION", 100))
NOISE_MAP_MARGIN = float(os.getenv("NOISE_MAP_MARGIN", 0.02))

def define_bounding_box(wind_turbines: dict) -> tuple:
    """Derive latitude and longitude axes for the simulation grid.

    :param wind_turbines: Turbine specifications including ``position`` tuples.
    :type wind_turbines: dict
    :returns: Tuple ``(latitudes, longitudes)`` defining the rectangular grid.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """

    # Determine the bounding box for the map

    lat_min = min(
        turbine["position"][0] for turbine in wind_turbines.values()
    )
    lat_max = max(
        turbine["position"][0] for turbine in wind_turbines.values()
    )
    lon_min = min(
        turbine["position"][1] for turbine in wind_turbines.values()
    )
    lon_max = max(
        turbine["position"][1] for turbine in wind_turbines.values()
    )

    # Adjust the map size to include observation points

    lat_min -= NOISE_MAP_MARGIN
    lat_max += NOISE_MAP_MARGIN
    lon_min -= NOISE_MAP_MARGIN
    lon_max += NOISE_MAP_MARGIN

    lon_array = np.linspace(lon_min, lon_max, NOISE_MAP_RESOLUTION)
    lat_array = np.linspace(lat_min, lat_max, NOISE_MAP_RESOLUTION)
    LON, LAT = np.meshgrid(lon_array, lat_array)

    return LAT[:, 0], LON[0, :]

class NoisePropagation:
    """Model sound propagation and attenuation for wind turbine layouts."""

    def __init__(
        self,
        wind_turbines: dict,
        humidity: float = 70,
        temperature: float = 20,
        elevation_data: xr.Dataset = None,
    ):
        """Initialise the propagation model and compute noise layers.

        :param wind_turbines: Turbine specifications including noise emission
            curves and mean wind speeds.
        :type wind_turbines: dict
        :param humidity: Relative humidity expressed as a percentage.
        :type humidity: float
        :param temperature: Air temperature in degrees Celsius.
        :type temperature: float
        :param elevation_data: Optional elevation dataset.
        :type elevation_data: xarray.DataArray | xarray.Dataset | None
        """
        self.incr_noise_att = None
        self.noise_attenuation = None
        self.elevation_grid = None
        self.haversine_distances = None
        if isinstance(humidity, dict) and temperature == 20 and elevation_data is None:
            listeners = humidity
            self.humidity = 70
            self.temperature = 20
        else:
            listeners = None
            self.temperature = temperature
            self.humidity = humidity
        self.listeners = listeners
        self.wind_turbines = wind_turbines
        self.LAT, self.LON = define_bounding_box(wind_turbines)
        self.elevation_data = elevation_data
        self.calculate_noise_attenuation_terms()

        self.noise_level_at_wind_speeds = self.get_noise_emissions_vs_time_or_speed(
            np.vstack(
                [
                    specs["noise_vs_wind_speed"].values
                    for specs in self.wind_turbines.values()
                ]
            ),
            coord_name="wind_speed",
            coord_value=[specs["noise_vs_wind_speed"].coords["wind_speed"].values for specs in self.wind_turbines.values()][0],
        )

        self.calculate_hourly_noise_levels()

        self.hourly_noise_levels = self.get_noise_emissions_vs_time_or_speed(
            np.vstack(
                [
                    specs["noise_per_hour"].values
                    for specs in self.wind_turbines.values()
                ]
            ),
            coord_name="hour",
            coord_value=[specs["noise_per_hour"].coords["hour"].values for specs in self.wind_turbines.values()][0],
        )

        self.l_den = self.compute_lden()
        self.l_night = self.compute_lnight()

        # Store the dataset as a class attribute
        self.incr_noise_att = self.calculate_incremental_noise_attenuation(self.l_den)
        self.incr_noise_att_night = self.calculate_incremental_noise_attenuation(self.l_night)


    def calculate_hourly_noise_levels(self):
        """Interpolate hourly emission levels for each turbine."""

        for turbine, turbine_specs in self.wind_turbines.items():
            wind_speeds = turbine_specs["mean_wind_speed"].values.flatten()
            noise_levels = turbine_specs["noise_vs_wind_speed"].values
            noise_level_wind_speeds = turbine_specs["noise_vs_wind_speed"].coords["wind_speed"].values

            # Create interpolation function
            interpolate_noise = interp1d(
                noise_level_wind_speeds,
                noise_levels,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Interpolate to find noise levels for the average wind speeds
            calculated_noise_levels = interpolate_noise(wind_speeds)

            # Create an xarray DataArray for the results
            noise_per_hour = xr.DataArray(
                calculated_noise_levels,
                dims=["hour"],
                coords={"hour": np.arange(len(wind_speeds))},
                name="noise_level"
            )

            # Add metadata
            noise_per_hour.attrs["units"] = "dB"
            noise_per_hour.attrs["description"] = "Predicted noise levels for hourly average wind speeds"

            # Add the noise levels to the wind turbine specs
            turbine_specs["noise_per_hour"] = noise_per_hour

    def compute_lden(self):
        """Calculate the day-evening-night noise level from hourly levels.

        :returns: Lden noise level raster.
        :rtype: xarray.DataArray
        """
        # Assuming the noise map has a 'time' coordinate for hourly noise levels
        noise = self.hourly_noise_levels  # Noise levels as DataArray

        # Define time ranges for day, evening, and night
        day_mask = (noise["hour"] >= 7) & (noise["hour"] < 19)  # 07:00–19:00
        evening_mask = (noise["hour"] >= 19) & (noise["hour"] < 23)  # 19:00–23:00
        night_mask = (noise["hour"] >= 23) | (noise["hour"] < 7)  # 23:00–07:00

        # Convert noise levels to linear scale and apply weightings
        day_linear = 10 ** (noise.where(day_mask).mean(dim="hour") / 10) * 12
        evening_linear = 10 ** ((noise.where(evening_mask).mean(dim="hour") + 5) / 10) * 4
        night_linear = 10 ** ((noise.where(night_mask).mean(dim="hour") + 10) / 10) * 8

        # Combine weighted intensities and compute Lden
        total_linear = (day_linear + evening_linear + night_linear) / 24
        lden = 10 * np.log10(total_linear)

        lden_map = noise.sel(hour=0).copy(data=lden)
        lden_map.attrs["long_name"] = "Lden Noise Levels"
        lden_map.attrs["units"] = "dB"

        return lden_map

    def compute_lnight(self):
        """Calculate the night noise level from hourly levels.

        :returns: Lnight noise level raster.
        :rtype: xarray.DataArray
        """
        noise = self.hourly_noise_levels  # Noise levels as DataArray

        # Define night time mask: 23:00–07:00
        night_mask = (noise["hour"] >= 23) | (noise["hour"] < 7)

        # Convert to linear scale, compute mean, then convert back to dB
        night_linear = 10 ** (noise.where(night_mask).mean(dim="hour") / 10)
        lnight = 10 * np.log10(night_linear)

        # Create DataArray with Lnight values
        lnight_map = noise.sel(hour=0).copy(data=lnight)
        lnight_map.attrs["long_name"] = "Lnight Noise Levels"
        lnight_map.attrs["units"] = "dB"

        return lnight_map

    def calculate_incremental_noise_attenuation(self, noise):
        """Apply successive attenuation terms to the emission raster.

        :param noise: Base noise level raster to which attenuations are applied.
        :type noise: xarray.DataArray
        :returns: Dataset containing intermediate attenuation stages.
        :rtype: xarray.Dataset
        """
        attenuation_terms = [
            "distance_attenuation",
            "atmospheric_absorption",
            "ground_attenuation",
            "obstacle_attenuation",
        ]

        short_names = {
            "distance_attenuation": "distance",
            "atmospheric_absorption": "atmospheric",
            "ground_attenuation": "ground",
            "obstacle_attenuation": "obstacle",
        }

        # Start with the initial noise level in linear scale
        incremental_noise_linear = 10 ** (noise / 10)

        # Create a dataset to store the incremental noise levels
        attenuation_dataset = xr.Dataset({"noise": noise.copy()})

        # Initialize cumulative attenuation in linear scale
        cumulative_attenuation_linear = xr.ones_like(incremental_noise_linear)

        for t, term in enumerate(attenuation_terms):
            if term in self.noise_attenuation.data_vars:
                label = "noise-" + "-".join([short_names[x] for x in attenuation_terms[: t + 1]])

                # Convert the current attenuation term to linear scale
                current_attenuation_linear = 10 ** (-self.noise_attenuation[term] / 10)

                # Combine the current attenuation with the cumulative attenuation
                cumulative_attenuation_linear *= current_attenuation_linear

                # Apply cumulative attenuation to the original noise
                attenuated_noise_linear = incremental_noise_linear * cumulative_attenuation_linear

                # Convert the result back to dB
                incremental_noise = 10 * np.log10(attenuated_noise_linear.clip(1e-10))  # Avoid log(0)

                # clip the noise level to 0 dB
                incremental_noise = np.clip(incremental_noise, a_min=0, a_max=None)

                # Add the resulting noise level to the dataset
                attenuation_dataset[label] = incremental_noise

        return attenuation_dataset


    def get_noise_emissions_vs_time_or_speed(self, noise, coord_name, coord_value) -> xr.DataArray:
        """Aggregate turbine emissions along the requested dimension.

        :param noise: Noise levels per turbine for each coordinate value.
        :type noise: numpy.ndarray
        :param coord_name: Name of the output coordinate (``"wind_speed"`` or
            ``"hour"``).
        :type coord_name: str
        :param coord_value: Coordinate values associated with the emissions.
        :type coord_value: numpy.ndarray
        :returns: Noise levels summed over all turbines for each grid cell.
        :rtype: xarray.DataArray
        """

        # we need to sum the noise levels along the turbine axis
        Z = 10 * np.log10((10 ** (noise / 10)).sum(axis=0))
        # resize Z to match the grid
        Z = np.tile(Z, (self.LAT.shape[0], self.LON.shape[0], 1))

        # create xarray to store Z
        Z = xr.DataArray(
            data=Z,
            dims=("lat", "lon", coord_name),
            coords={"lat": self.LAT, "lon": self.LON, coord_name: coord_value},
        )

        Z.values = np.clip(Z.values, a_min=0, a_max=None)

        return Z

    def noise_map_at_wind_speeds(self, noise):
        """Backward compatible alias for :meth:`get_noise_emissions_vs_time_or_speed`.

        :param noise: Noise levels per turbine for each wind speed.
        :type noise: numpy.ndarray
        :returns: Noise map aggregated across turbines for each wind speed.
        :rtype: xarray.DataArray
        """

        return self.get_noise_emissions_vs_time_or_speed(
            noise,
            coord_name="wind_speed",
            coord_value=[specs["noise_vs_wind_speed"].coords["wind_speed"].values for specs in self.wind_turbines.values()][0]
        )

    def calculate_sound_level_at_distance(self, *args, **kwargs):
        """Deprecated compatibility wrapper for legacy APIs.

        :raises NotImplementedError: Always raised to direct users to
            :meth:`get_noise_emissions_vs_time_or_speed`.
        """

        raise NotImplementedError(
            "calculate_sound_level_at_distance has been replaced by "
            "get_noise_emissions_vs_time_or_speed"
        )

    def superimpose_wind_turbines_noise(self):
        """Deprecated alias returning the aggregated wind speed noise map."""

        return self.noise_level_at_wind_speeds

    def calculate_noise_attenuation_terms(self):
        """Compute distance, atmospheric, ground and obstacle attenuation."""

        if getattr(self, "listeners", None) is not None:
            zeros = xr.DataArray(
                np.zeros((len(self.LAT), len(self.LON))),
                coords={"lat": self.LAT, "lon": self.LON},
                dims=["lat", "lon"],
            )
            self.noise_attenuation = xr.Dataset({
                "ground_attenuation": zeros,
                "obstacle_attenuation": zeros,
                "distance_attenuation": zeros,
                "atmospheric_absorption": zeros,
            })
            self.individual_noise = []
            return

        positions = [point["position"] for point in self.wind_turbines.values()]

        # Calculate the noise level or distances
        haversine_distances = np.array(
            [
                [
                    [
                        haversine(point1=(lat, lon), point2=position, unit=Unit.METERS)
                        for position in positions
                    ]
                    for lon in self.LON
                ]
                for lat in self.LAT
            ]
        )

        # Convert distances into an xarray.DataArray
        self.haversine_distances = xr.DataArray(
            haversine_distances,
            dims=("lat", "lon", "turbine"),
            coords={
                "lat": self.LAT,
                "lon": self.LON,
                "turbine": list(self.wind_turbines.keys())
            },
            name="haversine_distances"
        )

        # Calculate the geometric spreading loss, according to ISO 9613-2
        distance_attenuation = get_geometric_spread_loss(self.haversine_distances.values)
        # pick the minimum value along the turbine axis
        distance_attenuation = np.min(distance_attenuation, axis=-1)

        # Calculate the atmospheric absorption loss, according to ISO 9613-2
        atmospheric_absorption = (get_absorption_coefficient(
            self.temperature,
            self.humidity
        ) * self.haversine_distances.values) / 1000

        # pick the minimum value along the turbine axis
        atmospheric_absorption = np.min(atmospheric_absorption, axis=-1)

        # calculation elevation of grid cells compared to turbines' positions
        # we do this by subtracting the elevation of each grid cell from the elevation of the turbines
        self.elevation_grid, ground_attenuation, obstacle_attenuation = calculate_ground_attenuation(
            self.haversine_distances,
            self.LON,
            self.LAT,
            self.wind_turbines,
            self.elevation_data
        )

        data_vars = {}
        if ground_attenuation is not None:
            data_vars["ground_attenuation"] = (["lat", "lon"], ground_attenuation.data)
        if obstacle_attenuation is not None:
            data_vars["obstacle_attenuation"] = (["lat", "lon"], obstacle_attenuation.data)
        if distance_attenuation is not None:
            data_vars["distance_attenuation"] = (["lat", "lon"], distance_attenuation)
        if atmospheric_absorption is not None:
            data_vars["atmospheric_absorption"] = (["lat", "lon"], atmospheric_absorption)

        # Create an xarray Dataset
        self.noise_attenuation = xr.Dataset(
            data_vars,
            coords={
                "lat": self.LAT,
                "lon": self.LON,
            },
        )

    def plot_noise_map(self, dimension: str = "wind_speed"):
        """Display interactive contour plots of turbine noise levels.

        :param dimension: Dimension to explore (``"wind_speed"`` or ``"hour"``).
        :type dimension: str
        """

        # Create a wind speed slider for user interaction
        if dimension == "wind_speed":
            slider = widgets.FloatSlider(
                value=7.0,
                min=3.0,
                max=12.0,
                step=1.0,
                description="Wind Speed (m/s):",
                continuous_update=True,
            )
        else:
            # against hours of the day
            slider = widgets.IntSlider(
                value=12,
                min=0,
                max=23,
                step=1,
                description="Hour of the day:",
                continuous_update=True,
            )

        @widgets.interact(wind_speed=slider)
        def interactive_plot(wind_speed):
            plt.figure(figsize=(10, 6))

            if dimension == "wind_speed":
                data = self.noise_level_at_wind_speeds.interp(
                    wind_speed=wind_speed, kwargs={"fill_value": "extrapolate"}
                )
            else:
                data = self.hourly_noise_levels.interp(
                    hour=wind_speed, kwargs={"fill_value": "extrapolate"}
                )

            # Define contour levels starting from 35 dB
            contour_levels = [35, 40, 45, 50, 55, 60]

            # add bounding box
            plt.xlim(self.LON.min(), self.LON.max())
            plt.ylim(self.LAT.min(), self.LAT.max())

            plt.contourf(
                self.LON,  # x-axis, longitude
                self.LAT,  # y-axis, latitude
                data,
                levels=contour_levels,
                cmap="RdYlBu_r",
            )
            plt.colorbar(label="Noise Level (dB)")
            plt.title("Wind Turbine Noise Contours")
            plt.xlabel("Longitude")  # Correct label for x-axis
            plt.ylabel("Latitude")  # Correct label for y-axis

            # Plot wind turbines
            for turbine, specs in self.wind_turbines.items():
                plt.plot(
                    *specs["position"][::-1], "ko"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it, add a small offset to avoid overlapping
                plt.text(
                    specs["position"][1] + 0.003,
                    specs["position"][0] + 0.002,
                    turbine,
                )

            plt.grid(True)
            plt.show()

