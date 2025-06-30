import math
import json
from typing import Any

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from . import DATA_DIR
from .settlement import get_population_subset
from .noiseanalysis import NoiseAnalysis


def load_human_health_parameters() -> dict:
    """
    Load human health parameters from a JSON file.
    """
    with open(DATA_DIR / "health_parameters.json", "r") as f:
        return json.load(f)

def load_disease_data(country) -> Any | None:
    df = pd.read_csv(DATA_DIR / "data_diseases_europe.csv", sep=",")
    if country in df["country_short"].unique():
        return df.loc[df["country_short"] == country, :]
    else:
        print(
            f"No data available for country {country}. "
            f"Falling back to Europe"
        )
        return df.loc[df["country_long"] == "European Region", :]



def guess_country(bbox_geom) -> tuple[str, float]:
    """
    Returns the country (ISO_A2 code, POP_EST) corresponding to the bounding box geometry.
    """
    # Load world polygons
    world = gpd.read_file(DATA_DIR / "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    world = world.to_crs("EPSG:4326")  # Ensure CRS match

    # Clean geometries if needed
    world["geometry"] = world["geometry"].buffer(0)

    # Wrap bbox into a GeoDataFrame
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom.buffer(1e-4)], crs="EPSG:4326")

    # Compute intersection
    intersecting = gpd.overlay(world, bbox_gdf, how="intersection")

    # Skip if no matches
    if intersecting.empty:
        print("intersecting.empty")
        return "-99", 0

    # Reproject to compute area
    intersecting_proj = intersecting.to_crs("EPSG:3035")
    intersecting["area"] = intersecting_proj.geometry.area

    # Get largest intersection
    country = intersecting.sort_values("area", ascending=False).iloc[0]

    return country.get("ISO_A2_EH"), country.get("POP_EST", 0)


class HumanHealth:
    def __init__(self, noiseanalysis: NoiseAnalysis, lifetime : int = 20):
        self.l_den = noiseanalysis.l_den
        self.l_night = noiseanalysis.l_night
        self.lifetime = lifetime

        bbox = box(
            float(noiseanalysis.ambient_noise_map.coords["lon"].min()),
            float(noiseanalysis.ambient_noise_map.coords["lat"].min()),
            float(noiseanalysis.ambient_noise_map.coords["lon"].max()),
            float(noiseanalysis.ambient_noise_map.coords["lat"].max()),
        )

        self.population = get_population_subset(bbox).interp(
            lat=self.l_den.lat,
            lon=self.l_den.lon,
            method="linear"
        )

        self.country, self.country_population = guess_country(bbox)
        self.disease_data = load_disease_data(self.country)
        self.human_health_parameters = load_human_health_parameters()
        self.population_rate = self.population / self.country_population
        self.human_health = None
        self.calculate_total_dalys()


    def get_disease_totals(
        self,
        disease: str,
    ) -> tuple[float, float]:
        """
        Returns YLD and YLL for a given country and disease.
        :param disease: Name of the disease (e.g., "ischemic_heart_disease", "diabetes", "stroke").
        :return: Tuple of YLD and YLL values for the disease in the specified country.
        :raises ValueError: If no data is found for the specified disease in the country.
        """
        row = self.disease_data.loc[
            self.disease_data["disease"].str.lower() == disease.lower()
        ]
        if row.empty:
            raise ValueError(f"No data found for {disease} in {self.country}")
        return float(row["YLD"].values[0]), float(row["YLL"].values[0])


    def calculate_highly_annoyed_dalys(
        self,
        noise_type_ha: str
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for highly annoyed individuals
        based on the noise level (Lden) and the number of exposed individuals.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type_ha: Type of noise (e.g., "road_without_alpinestudies", "combined").
        :return: DALYs for highly annoyed individuals.
        """
        p = self.human_health_parameters["highly_annoyed"][noise_type_ha]
        a = p["a"]
        b = p["b"]
        c = p["c"]
        disability_weight = p["disability_weight"]
        percentage_affected = a + b * self.l_den + c * (self.l_den ** 2)
        affected = self.population * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime

        return dalys


    # 2. High Sleep Disorder
    def calculate_high_sleep_disorder_dalys(
        self,
        noise_type_hsd: str
    )-> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for high sleep disorder
        based on the noise level (Lnight) and the number of exposed individuals.
        :param lnight: Noise level in Lnight (night noise level).
        :param noise_type_hsd: Type of noise (e.g., "combined", "road_without_alpinestudies").
        :return: DALYs for high sleep disorder.
        """
        p = self.human_health_parameters["high_sleep_disorder"][noise_type_hsd]
        a = p["a"]
        b = p["b"]
        c = p["c"]
        disability_weight = p["disability_weight"]
        percentage_affected = a + b * self.l_night + c * (self.l_night ** 2)
        affected = self.population * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime

        return dalys


    # 3. Ischemic Heart Disease (IHD)
    def calculate_ihd_dalys(
        self,
        noise_type: str,
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for ischemic heart disease
        based on the noise level (Lden) and the number of exposed individuals.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for ischemic heart disease.
        """
        p = self.human_health_parameters["ischemic_heart_disease"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        rr = np.exp((np.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("ischemic_heart_disease")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            self.l_den >= threshold,
            dalys,
            0
        )

        return dalys


    # 4. Diabetes
    def calculate_diabetes_dalys(
        self,
        noise_type: str,
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for diabetes
        based on the noise level (Lden) and the number of exposed individuals.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for diabetes.
        """
        p = self.human_health_parameters["diabetes"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        rr = np.exp((np.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("diabetes")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            self.l_den >= threshold,
            dalys,
            0
        )

        return dalys


    def calculate_stroke_dalys(
            self,
            noise_type: str,
    ):
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for stroke
        based on the noise level (Lden) and the number of exposed individuals.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for stroke.
        """
        p = self.human_health_parameters["stroke"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        rr = np.exp((np.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("stroke")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            self.l_den >= threshold,
            dalys,
            0
        )

        return dalys


    def calculate_total_dalys(
        self,
        noise_type_ha: str = "road_without_alpinestudies",
        noise_type_hsd: str = "combined",
        noise_type: str = "road middle",
    ):
        """
        Calculate the total Disability-Adjusted Life Years (DALYs) for all health impacts
        based on the noise levels (Lden, Lnight) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
        :param noise_type_ha: Type of noise for highly annoyed (e.g., "road_without_alpinestudies", "combined").
        :param noise_type_hsd: Type of noise for high sleep disorder (e.g., "combined", "road_without_alpinestudies").
        :param noise_type: Type of noise for ischemic heart disease, diabetes, and stroke (e.g., "road middle", "railway").
        :return: Total DALYs for all health impacts.
        """
        dalys_ha = self.calculate_highly_annoyed_dalys(noise_type_ha)
        dalys_hsd = self.calculate_high_sleep_disorder_dalys(noise_type_hsd)
        dalys_ihd = self.calculate_ihd_dalys(noise_type)
        dalys_diabetes = self.calculate_diabetes_dalys(noise_type)
        dalys_stroke = self.calculate_stroke_dalys(noise_type)

        ds = xr.Dataset(
            {
                "highly_annoyed": dalys_ha,
                "high_sleep_disorder": dalys_hsd,
                "ischemic_heart_disease": dalys_ihd,
                "diabetes": dalys_diabetes,
                "stroke": dalys_stroke,
            }
        )

        # replace NaNs with zeroe
        ds = ds.fillna(0)

        self.human_health = ds
