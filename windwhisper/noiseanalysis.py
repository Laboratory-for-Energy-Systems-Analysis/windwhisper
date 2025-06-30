import numpy as np
import xarray as xr
from geojson import FeatureCollection
from typing import Any

from .ambient_noise import get_ambient_noise_levels
from .plotting import generate_map, create_geojson


class NoiseAnalysis:
    """
    This class handles the basic functionalities related to noise data analysis.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise_propagation: A NoiseMap object containing the noise data.

    """

    def __init__(
            self,
            noise_propagation,
            wind_turbines,
    ):
        self.noise_propagation = noise_propagation
        self.wind_turbines = wind_turbines
        self.l_den = noise_propagation.l_den
        self.l_night = noise_propagation.l_night

        self.ambient_noise_map = get_ambient_noise_levels(
            latitudes=self.noise_propagation.LAT,
            longitudes=self.noise_propagation.LON,
            resolution=self.l_den.shape
        )


        self.merged_map = self.merge_maps()

    def merge_maps(self):
        """
        Merge the ambient noise, Lden, and settlement maps into a single xarray dataset.
        """

        lon, lat = self.l_den.lon, self.l_den.lat

        # reinterpolate the ambient noise map to match the shape of the lden map
        self.ambient_noise_map = self.ambient_noise_map.interp(
            lon=lon,
            lat=lat
        ).fillna(0)

        # Combine the two datasets into a single xarray
        merged_dataset = xr.Dataset({
            "ambient": self.ambient_noise_map,
            "wind": self.noise_propagation.incr_noise_att["noise-distance-atmospheric-ground-obstacle"],
        })

        # Calculate the combined noise level (in dB)
        # using the logarithmic formula
        noise_combined = 10 * np.log10(
            10 ** (self.ambient_noise_map.values / 10)
            + 10 ** (self.noise_propagation.incr_noise_att["noise-distance-atmospheric-ground-obstacle"].values / 10)
        )

        # Add the new layer to the dataset
        merged_dataset["combined"] = xr.DataArray(
            noise_combined,
            dims=["lat", "lon"],
            coords={
                "lat": self.l_den.lat,
                "lon": self.l_den.lon,
            }
        )

        # Add metadata for clarity
        merged_dataset["combined"].attrs["description"] = "Combined noise levels (ambient + LDEN) in dB"
        merged_dataset["combined"].attrs["units"] = "dB"

        # Calculate the net contribution of lden_noise to the combined noise level
        net_contribution = 10 * np.log10(
            10 ** (noise_combined / 10) / 10 ** (self.ambient_noise_map.values / 10)
        )

        # Add the net contribution layer to the dataset
        merged_dataset["net"] = xr.DataArray(
            net_contribution,
            dims=["lat", "lon"],
            coords={
                "lat": self.l_den.coords["lat"],
                "lon": self.l_den.coords["lon"],
            }
        )
        merged_dataset["net"].attrs["description"] = "Net contribution of LDEN noise levels in dB"

        mask = (self.ambient_noise_map.values < 55) & (noise_combined >= 55)
        flip = np.where(mask, noise_combined, 0)

        # Add the flip layer to the dataset
        merged_dataset["flip"] = xr.DataArray(
            flip,
            dims=["lat", "lon"],
            coords={
                "lat": self.l_den.coords["lat"],
                "lon": self.l_den.coords["lon"],
            }
        )

        # Add metadata for clarity
        merged_dataset["flip"].attrs["description"] = (
            "Coordinates where ambient noise < 55 dB but combined noise > 55 dB"
        )
        merged_dataset["flip"].attrs["datatype"] = "boolean"

        return merged_dataset

    def generate_map(self, filepath="noise_map.html"):
        generate_map(
            noise_dataset=self.merged_map,
            filepath=filepath
        )


    def get_geojson_contours(self) -> tuple[Any, Any, Any, Any, Any]:
        """
        Generate a GeoJSON object containing the contours of the noise levels.
        """
        return (
            create_geojson(self.merged_map["ambient"]),
            create_geojson(self.merged_map["wind"]),
            create_geojson(self.merged_map["combined"]),
            create_geojson(self.merged_map["net"]),
            create_geojson(self.merged_map["flip"]),
        )


