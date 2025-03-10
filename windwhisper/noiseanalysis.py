import numpy as np
import xarray as xr
from .ambient_noise import get_ambient_noise_levels
from .settlement import get_wsf_map_preview
from .plotting import generate_map


class NoiseAnalysis:
    """
    This class handles the basic functionalities related to noise data analysis.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise_propagation: A NoiseMap object containing the noise data.

    """

    def __init__(self, noise_propagation, wind_turbines):
        self.noise_propagation = noise_propagation
        self.wind_turbines = wind_turbines
        self.lden_map = noise_propagation.lden_map

        lon_min = self.noise_propagation.hourly_noise_levels.coords["lon"].min().values.item()
        lon_max = self.noise_propagation.hourly_noise_levels.coords["lon"].max().values.item()
        lat_min = self.noise_propagation.hourly_noise_levels.coords["lat"].min().values.item()
        lat_max = self.noise_propagation.hourly_noise_levels.coords["lat"].max().values.item()

        self.ambient_noise_map = get_ambient_noise_levels(
            latitudes=self.noise_propagation.LAT,
            longitudes=self.noise_propagation.LON,
            resolution=self.lden_map.shape
        )

        #self.settlement_map = get_wsf_map_preview(
        #    lon_min=lon_min,
        #    lon_max=lon_max,
        #    lat_min=lat_min,
        #    lat_max=lat_max,
        #    resolution=self.lden_map.shape
        #)
        self.settlement_map = None

        self.merged_map = self.merge_maps()



    def merge_maps(self):
        """
        Merge the ambient noise, Lden, and settlement maps into a single xarray dataset.
        """

        lon, lat = self.lden_map.lon, self.lden_map.lat

        # reinterpolate the ambient noise map to match the shape of the lden map
        self.ambient_noise_map = self.ambient_noise_map.interp(
            lon=lon,
            lat=lat
        ).fillna(0)

        if self.settlement_map:
            # re-interpolate the settlement map to match the shape of the lden map
            self.settlement_map = self.settlement_map.interp(
                lon=lon,
                lat=lat
            ).fillna(0)

        # Combine the two datasets into a single xarray
        merged_dataset = xr.Dataset({
            "ambient": self.ambient_noise_map,
            "wind": self.noise_propagation.incr_noise_att["noise-distance-atmospheric-ground-obstacle"],
            #"settlement": self.settlement_map,
        })

        if self.settlement_map:
            merged_dataset["settlement"] = self.settlement_map

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
                "lat": self.lden_map.lat,
                "lon": self.lden_map.lon,
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
                "lat": self.lden_map.coords["lat"],
                "lon": self.lden_map.coords["lon"],
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
                "lat": self.lden_map.coords["lat"],
                "lon": self.lden_map.coords["lon"],
            }
        )

        # Add metadata for clarity
        merged_dataset["flip"].attrs["description"] = (
            "Coordinates where ambient noise < 55 dB but combined noise > 55 dB"
        )
        merged_dataset["flip"].attrs["datatype"] = "boolean"

        return merged_dataset

    def generate_map(self):
        generate_map(self.merged_map)


