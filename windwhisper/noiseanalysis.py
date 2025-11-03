import numpy as np
import xarray as xr
from typing import Any

from .ambient_noise import get_ambient_noise_levels
from .plotting import generate_map, create_geojson


class NoiseAnalysis:
    """Aggregate noise propagation, ambient data and outputs for reporting."""

    def __init__(
            self,
            noise_propagation,
            wind_turbines,
            listeners=None,
    ):
        """Initialise the noise analysis from propagation results.

        :param noise_propagation: Propagation model containing incremental
            noise layers.
        :type noise_propagation: windwhisper.noisepropagation.NoisePropagation
        :param wind_turbines: Turbine specifications keyed by identifier.
        :type wind_turbines: dict
        :param listeners: Optional listener definitions kept for backwards
            compatibility.
        :type listeners: dict | None
        """
        self.noise_propagation = noise_propagation
        self.wind_turbines = wind_turbines
        self.l_den = noise_propagation.l_den
        self.l_night = noise_propagation.l_night

        if not isinstance(self.l_den, xr.DataArray) or not isinstance(self.l_night, xr.DataArray):
            default_lat = np.array([0.0])
            default_lon = np.array([0.0])
            placeholder = xr.DataArray(
                np.zeros((len(default_lat), len(default_lon))),
                dims=["lat", "lon"],
                coords={"lat": default_lat, "lon": default_lon},
                name="noise"
            )
            self.l_den = placeholder
            self.l_night = placeholder.copy()
            self.noise_propagation.LAT = default_lat
            self.noise_propagation.LON = default_lon
            self.noise_propagation.incr_noise_att = xr.Dataset({
                "noise-distance-atmospheric-ground-obstacle": placeholder
            })
            self.noise_propagation.incr_noise_att_night = xr.Dataset({
                "noise-distance-atmospheric-ground-obstacle": placeholder
            })

        try:
            latitudes = getattr(self.noise_propagation, "LAT", self.l_den.coords["lat"].values)
            longitudes = getattr(self.noise_propagation, "LON", self.l_den.coords["lon"].values)
            if not hasattr(latitudes, "min"):
                latitudes = self.l_den.coords["lat"].values
            if not hasattr(longitudes, "min"):
                longitudes = self.l_den.coords["lon"].values

            self.ambient_noise_map_lden = get_ambient_noise_levels(
                latitudes=latitudes,
                longitudes=longitudes,
                resolution=self.l_den.shape,
            )

            self.ambient_noise_map_lnight = get_ambient_noise_levels(
                latitudes=latitudes,
                longitudes=longitudes,
                resolution=self.l_den.shape,
                lden=False
            )
        except Exception:
            self.ambient_noise_map_lden = self.l_den.copy(data=np.zeros(self.l_den.shape))
            self.ambient_noise_map_lnight = self.l_night.copy(data=np.zeros(self.l_night.shape))

        self.merged_map, self.merged_map_night = self.merge_maps()

    def merge_maps(self):
        """Merge ambient and propagation rasters into combined datasets.

        :returns: Tuple ``(merged_dataset, merged_dataset_night)`` containing
            Lden and Lnight datasets with ambient, wind, combined, net and flip
            layers.
        :rtype: tuple[xarray.Dataset, xarray.Dataset]
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

        noise_combined = np.maximum(noise_combined, self.ambient_noise_map_lden.values)

        noise_combined_night = 10 * np.log10(
            10 ** (self.ambient_noise_map_lnight.values / 10)
            + 10 ** (self.noise_propagation.incr_noise_att_night["noise-distance-atmospheric-ground-obstacle"].values / 10)
        )

        noise_combined_night = np.maximum(noise_combined_night, self.ambient_noise_map_lnight.values)

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
        """Render interactive HTML maps for Lden and Lnight rasters.

        :param filepath: Base filepath for the generated HTML maps.
        :type filepath: str
        """
        generate_map(
            noise_dataset=self.merged_map,
            filepath=f"{filepath.replace('.html', '_lden.html')}",
        )

        generate_map(
            noise_dataset=self.merged_map_night,
            filepath=f"{filepath.replace('.html', '_lnight.html')}",
        )


    def get_geojson_contours(self) -> tuple[Any, Any, Any, Any, Any]:
        """Create GeoJSON contour layers for the combined noise outputs.

        :returns: Tuple containing GeoJSON objects for daytime and night-time
            combined, ambient and net layers plus the coordinate reference
            system string.
        :rtype: tuple[Any, Any, Any, Any, Any]
        """
        return (
            create_geojson(self.merged_map["ambient"]),
            create_geojson(self.merged_map["wind"]),
            create_geojson(self.merged_map["combined"]),
            create_geojson(self.merged_map["net"]),
            create_geojson(self.merged_map["flip"]),
        )


