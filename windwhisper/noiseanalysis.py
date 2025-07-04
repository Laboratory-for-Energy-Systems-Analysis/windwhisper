import numpy as np
import xarray as xr
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

        self.ambient_noise_map_lden = get_ambient_noise_levels(
            latitudes=self.noise_propagation.LAT,
            longitudes=self.noise_propagation.LON,
            resolution=self.l_den.shape,
        )

        self.ambient_noise_map_lnight = get_ambient_noise_levels(
            latitudes=self.noise_propagation.LAT,
            longitudes=self.noise_propagation.LON,
            resolution=self.l_den.shape,
            lden=False
        )

        self.merged_map, self.merged_map_night = self.merge_maps()

    def merge_maps(self):
        """
        Merge the ambient noise, Lden, and settlement maps into a single xarray dataset.
        """

        lon, lat = self.l_den.coords["lon"].values, self.l_den.coords["lat"].values

        # reinterpolate the ambient noise map to match the shape of the lden map
        self.ambient_noise_map_lden = self.ambient_noise_map_lden.interp(
            lon=lon,
            lat=lat
        ).fillna(0)

        self.ambient_noise_map_lnight = self.ambient_noise_map_lnight.interp(
            lon=lon,
            lat=lat
        ).fillna(0)

        # Combine the two datasets into a single xarray
        merged_dataset = xr.Dataset({
            "ambient": self.ambient_noise_map_lden,
            "wind": self.noise_propagation.incr_noise_att["noise-distance-atmospheric-ground-obstacle"],
        })

        merged_dataset_night = xr.Dataset({
            "ambient": self.ambient_noise_map_lnight,
            "wind": self.noise_propagation.incr_noise_att_night["noise-distance-atmospheric-ground-obstacle"],
        })

        # Calculate the combined noise level (in dB)
        # using the logarithmic formula
        noise_combined = 10 * np.log10(
            10 ** (self.ambient_noise_map_lden.values / 10)
            + 10 ** (self.noise_propagation.incr_noise_att["noise-distance-atmospheric-ground-obstacle"].values / 10)
        )

        noise_combined_night = 10 * np.log10(
            10 ** (self.ambient_noise_map_lnight.values / 10)
            + 10 ** (self.noise_propagation.incr_noise_att_night["noise-distance-atmospheric-ground-obstacle"].values / 10)
        )

        # Add the new layer to the dataset
        merged_dataset["combined"] = xr.DataArray(
            noise_combined,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )

        merged_dataset_night["combined"] = xr.DataArray(
            noise_combined_night,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )

        # Add metadata for clarity
        merged_dataset["combined"].attrs["description"] = "Combined noise levels (ambient + LDEN) in dB"
        merged_dataset["combined"].attrs["units"] = "dB"

        merged_dataset_night["combined"].attrs["description"] = "Combined noise levels (ambient + LNIGHT) in dB"
        merged_dataset_night["combined"].attrs["units"] = "dB"

        # Calculate the net contribution of lden_noise to the combined noise level
        net_contribution = 10 * np.log10(
            10 ** (noise_combined / 10) / 10 ** (self.ambient_noise_map_lden.values / 10)
        )

        net_contribution_night = 10 * np.log10(
            10 ** (noise_combined_night / 10) / 10 ** (self.ambient_noise_map_lnight.values / 10)
        )

        # Add the net contribution layer to the dataset
        merged_dataset["net"] = xr.DataArray(
            net_contribution,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )
        merged_dataset["net"].attrs["description"] = "Net contribution of LDEN noise levels in dB"

        merged_dataset_night["net"] = xr.DataArray(
            net_contribution_night,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )
        merged_dataset_night["net"].attrs["description"] = "Net contribution of LDEN noise levels in dB, at night"

        mask = (self.ambient_noise_map_lden.values < 55) & (noise_combined >= 55)
        flip = np.where(mask, noise_combined, 0)

        # Add the flip layer to the dataset
        merged_dataset["flip"] = xr.DataArray(
            flip,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )

        # Add metadata for clarity
        merged_dataset["flip"].attrs["description"] = (
            "Coordinates where ambient noise < 55 dB but combined noise > 55 dB"
        )
        merged_dataset["flip"].attrs["datatype"] = "boolean"

        mask = (self.ambient_noise_map_lnight.values < 50) & (noise_combined_night >= 50)
        flip = np.where(mask, noise_combined_night, 0)

        # Add the flip layer to the dataset
        merged_dataset_night["flip"] = xr.DataArray(
            flip,
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            }
        )

        # Add metadata for clarity
        merged_dataset_night["flip"].attrs["description"] = (
            "Coordinates where ambient noise < 50 dB but combined noise > 50 dB"
        )
        merged_dataset_night["flip"].attrs["datatype"] = "boolean"

        return merged_dataset, merged_dataset_night

    def generate_map(self, filepath="noise_map.html"):
        generate_map(
            noise_dataset=self.merged_map,
            filepath=f"{filepath.replace('.html', '_lden.html')}",
        )

        generate_map(
            noise_dataset=self.merged_map_night,
            filepath=f"{filepath.replace('.html', '_lnight.html')}",
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


