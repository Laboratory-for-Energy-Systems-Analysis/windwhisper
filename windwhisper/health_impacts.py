import json
from typing import Any

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from pyproj import Geod

from . import DATA_DIR
from .settlement import get_population_subset
from .noiseanalysis import NoiseAnalysis
from .electricity_production import get_electricity_production


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


geod = Geod(ellps="WGS84")

def approximate_grid_cell_areas(lat, lon):
    """
    Compute area of each lat-lon cell (in m²) for a rectilinear grid.
    Inputs:
        lat (1D array): latitude coordinates
        lon (1D array): longitude coordinates
    Output:
        area (2D array): (lat, lon) grid of cell areas in m²
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    dlat = np.diff(lat).mean()
    dlon = np.diff(lon).mean()

    lat_edges = np.concatenate([
        [lat[0] - dlat / 2],
        (lat[:-1] + lat[1:]) / 2,
        [lat[-1] + dlat / 2]
    ])
    lon_edges = np.concatenate([
        [lon[0] - dlon / 2],
        (lon[:-1] + lon[1:]) / 2,
        [lon[-1] + dlon / 2]
    ])

    area = np.zeros((len(lat), len(lon)))

    for i in range(len(lat)):
        for j in range(len(lon)):
            # bounding box of each cell
            l, r = lon_edges[j], lon_edges[j + 1]
            b, t = lat_edges[i + 1], lat_edges[i]
            # area in m²
            poly_area, _ = geod.polygon_area_perimeter([l, r, r, l, l], [b, b, t, t, b])
            area[i, j] = abs(poly_area)

    return area  # shape (lat, lon)

class HumanHealth:
    def __init__(self, noiseanalysis: NoiseAnalysis, lifetime : int = 20):
        #self.l_den = noiseanalysis.merged_map["combined"]
        #self.l_night = noiseanalysis.merged_map_night["combined"]
        self.lifetime = lifetime
        self.electricity_production = sum(
            get_electricity_production(
                lat=turbine["position"][0],
                lon=turbine["position"][1],
                power=turbine["power"],
                lifetime=lifetime
            ) for turbine in noiseanalysis.wind_turbines.values()
        )

        bbox = box(
            float(noiseanalysis.ambient_noise_map_lden.coords["lon"].min()),
            float(noiseanalysis.ambient_noise_map_lden.coords["lat"].min()),
            float(noiseanalysis.ambient_noise_map_lden.coords["lon"].max()),
            float(noiseanalysis.ambient_noise_map_lden.coords["lat"].max()),
        )

        # 1. Get original population data (people per cell)
        source = get_population_subset(bbox)  # (lat, lon) grid

        # 2. Compute original grid cell areas
        area_src = approximate_grid_cell_areas(source.lat.values, source.lon.values)
        pop_density = source / area_src  # people/m²

        # 3. Interpolate to target grid (same units: people/m²)
        pop_density_interp = pop_density.interp(
            lat=noiseanalysis.ambient_noise_map_lden.coords["lat"].values,
            lon=noiseanalysis.ambient_noise_map_lden.coords["lon"].values,
            method="linear"
        )

        # 4. Compute area of target grid cells
        area_dst = approximate_grid_cell_areas(
            noiseanalysis.ambient_noise_map_lden.coords["lat"].values,
            noiseanalysis.ambient_noise_map_lden.coords["lon"].values
        )

        # 5. Multiply to get people per cell
        self.population = (pop_density_interp * area_dst).astype("float32")

        self.noiseanalysis = noiseanalysis
        self.country, self.country_population = guess_country(bbox)
        self.disease_data = load_disease_data(self.country)
        self.human_health_parameters = load_human_health_parameters()
        self.population_rate = self.population / self.country_population

        self.human_health_wo_turbines = self.calculate_total_dalys(
            lden=noiseanalysis.merged_map["ambient"],
            lnight=noiseanalysis.merged_map_night["ambient"]
        )
        self.human_health_per_kWh_wo_turbines = self.human_health_wo_turbines / self.electricity_production

        self.human_health = self.calculate_total_dalys(
            lden=noiseanalysis.merged_map["combined"],
            lnight=noiseanalysis.merged_map["combined"]
        )
        self.human_health_per_kWh = self.human_health / self.electricity_production


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
            lden: xr.DataArray,
            noise_type_ha: str
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for highly annoyed individuals
        based on the noise level (Lden) and the number of exposed individuals.
        Applies a correction to avoid increasing DALYs at low Lden due to the shape of the quadratic fit.

        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type_ha: Type of noise (e.g., "road_without_alpinestudies", "combined").
        :return: DALYs for highly annoyed individuals.
        """
        # Load parameters
        p = self.human_health_parameters["highly_annoyed"][noise_type_ha]
        a = p["a"]
        b = p["b"]
        c = p["c"]
        threshold = 0
        disability_weight = p["disability_weight"]

        # Compute Lden value at minimum of the quadratic function
        lden_min = -b / (2 * c)
        min_percentage = a + b * lden_min + c * (lden_min ** 2)

        # Calculate percentage affected,
        # with correction below the minimum
        percentage_affected = xr.where(
            lden < lden_min,
            min_percentage,
            a + b * lden + c * (lden ** 2)
        )

        # Compute DALYs
        affected = self.population * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            lden >= threshold,
            dalys,
            0
        )

        return dalys

    # 2. High Sleep Disorder
    def calculate_high_sleep_disorder_dalys(
            self,
            lnight: xr.DataArray,
            noise_type_hsd: str
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for high sleep disorder
        based on the noise level (Lnight) and the number of exposed individuals.
        Applies a correction to avoid increasing DALYs at low Lnight due to the shape of the quadratic fit.

        :param lnight: Noise level in Lnight (night noise level).
        :param noise_type_hsd: Type of noise (e.g., "combined", "road_without_alpinestudies").
        :return: DALYs for high sleep disorder.
        """
        # Load parameters
        p = self.human_health_parameters["high_sleep_disorder"][noise_type_hsd]
        a = p["a"]
        b = p["b"]
        c = p["c"]

        threshold = 0
        disability_weight = p["disability_weight"]

        # Compute Lnight value at minimum of the quadratic function
        lnight_min = -b / (2 * c)
        min_percentage = a + b * lnight_min + c * (lnight_min ** 2)

        # Calculate percentage affected, with correction below the minimum
        percentage_affected = xr.where(
            lnight < lnight_min,
            min_percentage,
            a + b * lnight + c * (lnight ** 2)
        )

        # Compute DALYs
        affected = self.population * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            lnight >= threshold,
            dalys,
            0
        )

        return dalys

    # 3. Ischemic Heart Disease (IHD)
    def calculate_ihd_dalys(
        self,
        lden: xr.DataArray,
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

        rr = np.exp((np.log(rr_per_10db) / 10) * (lden - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("ischemic_heart_disease")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            lden >= threshold,
            dalys,
            0
        )

        return dalys


    # 4. Diabetes
    def calculate_diabetes_dalys(
        self,
        lden: xr.DataArray,
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

        rr = np.exp((np.log(rr_per_10db) / 10) * (lden - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("diabetes")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            lden >= threshold,
            dalys,
            0
        )

        return dalys


    def calculate_stroke_dalys(
        self,
        lden: xr.DataArray,
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

        rr = np.exp((np.log(rr_per_10db) / 10) * (lden - threshold))
        paf = self.population_rate * (rr - 1) / (self.population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("stroke")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        # zero DALYs were value below threshold
        dalys = xr.where(
            lden >= threshold,
            dalys,
            0
        )

        return dalys


    def calculate_total_dalys(
        self,
        lden: xr.DataArray,
        lnight: xr.DataArray,
        noise_type_ha: str = "road_without_alpinestudies",
        noise_type_hsd: str = "combined",
        noise_type: str = "road middle",
    ) -> xr.Dataset:
        """
        Calculate the total Disability-Adjusted Life Years (DALYs) for all health impacts
        based on the noise levels (Lden, Lnight) and the number of exposed individuals.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param lnight: Noise level in Lnight (night noise level).
        :param noise_type_ha: Type of noise for highly annoyed (e.g., "road_without_alpinestudies", "combined").
        :param noise_type_hsd: Type of noise for high sleep disorder (e.g., "combined", "road_without_alpinestudies").
        :param noise_type: Type of noise for ischemic heart disease, diabetes, and stroke (e.g., "road middle", "railway").
        :return: Total DALYs for all health impacts.
        """
        dalys_ha = self.calculate_highly_annoyed_dalys(lden, noise_type_ha)
        dalys_hsd = self.calculate_high_sleep_disorder_dalys(lnight, noise_type_hsd)
        dalys_ihd = self.calculate_ihd_dalys(lden, noise_type)
        dalys_diabetes = self.calculate_diabetes_dalys(lden, noise_type)
        dalys_stroke = self.calculate_stroke_dalys(lden, noise_type)

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

        return ds

    def export_to_excel(self, filepath: str = "human_health_results.xlsx") -> None:
        """
        Export summary and raster data to an Excel file with multiple sheets:
        - Sheet 1: Metadata (country, population, disease data, parameters)
        - Sheet 2: L_den (100x100)
        - Sheet 3: L_night (100x100)
        - Sheet 4+: DALY layers (100x100 per disease cause)
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # --- Sheet 1: Metadata summary ---
            metadata = {
                "Country": [self.country],
                "Estimated population": [self.country_population],
                "Population in area": [self.population.sum().item()],
                "Lifetime (years) of wind turbines": [self.lifetime],
                "Number of wind turbines": [len(self.noiseanalysis.wind_turbines)],
                "Power (kW) of wind turbines": [turbine["power"] for turbine in self.noiseanalysis.wind_turbines.values()],
                "Location of wind turbines (lat, lon)": [
                    f"({turbine['position'][0]}, {turbine['position'][1]})"
                    for turbine in self.noiseanalysis.wind_turbines.values()
                ],
                "Bounding box (lat , lon)": [
                    f"({self.noiseanalysis.ambient_noise_map_lden.coords['lat'].min().item(0)}, {self.noiseanalysis.ambient_noise_map_lden.coords['lon'].min().item(0)}) - "
                    f"({self.noiseanalysis.ambient_noise_map_lden.coords['lat'].max().item(0)}, {self.noiseanalysis.ambient_noise_map_lden.coords['lon'].max().item(0)}) - "
                ],
                "Electricity production (kWh) over lifetime": [self.electricity_production],
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name="summary", index=False)

            # Append disease data (flattened)
            disease_flat = self.disease_data.reset_index(drop=True)
            disease_flat.to_excel(writer, sheet_name="summary", startrow=4, index=False)

            # Append health parameters (flattened JSON)
            hp_flat = pd.json_normalize(self.human_health_parameters, sep="_")
            hp_flat.T.reset_index().rename(columns={"index": "parameter", 0: "value"}) \
                .to_excel(writer, sheet_name="summary", startrow=6 + len(disease_flat), index=False)

            # --- Sheet 2: L_den without turbines ---
            df_ld = pd.DataFrame(
                self.noiseanalysis.merged_map["ambient"].values,
                index=self.noiseanalysis.merged_map["ambient"].coords["lat"].values,
                columns=self.noiseanalysis.merged_map["ambient"].coords["lon"].values
            )
            df_ld.to_excel(writer, sheet_name="L_den_wo_turb")

            # --- Sheet 3: L_night without turbines ---
            df_ln = pd.DataFrame(
                self.noiseanalysis.merged_map_night["ambient"].values,
                index=self.noiseanalysis.merged_map_night["ambient"].coords["lat"].values,
                columns=self.noiseanalysis.merged_map_night["ambient"].coords["lon"].values
            )
            df_ln.to_excel(writer, sheet_name="L_night_wo_turb")

            # --- Sheet 2: L_den with turbines ---
            df_ld = pd.DataFrame(
                self.noiseanalysis.merged_map["combined"].values,
                index=self.noiseanalysis.merged_map["combined"].coords["lat"].values,
                columns=self.noiseanalysis.merged_map["combined"].coords["lon"].values
            )
            df_ld.to_excel(writer, sheet_name="L_den_w_turb")

            # --- Sheet 3: L_night with turbines ---
            df_ln = pd.DataFrame(
                self.noiseanalysis.merged_map_night["combined"].values,
                index=self.noiseanalysis.merged_map_night["combined"].coords["lat"].values,
                columns=self.noiseanalysis.merged_map_night["combined"].coords["lon"].values
            )
            df_ln.to_excel(writer, sheet_name="L_night_w_turb")

            # --- Sheet 4: population ---
            df_pop = pd.DataFrame(self.population.values, index=self.population.lat.values, columns=self.population.lon.values)
            df_pop.to_excel(writer, sheet_name="Population")

            # --- Sheets 5+: DALYs per impact ---
            for varname in self.human_health_wo_turbines.data_vars:
                data = self.human_health_wo_turbines[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_wo_turb")

            # --- Final Sheet: Total DALYs (sum of all layers) ---
            total_dalys = self.human_health_wo_turbines.to_array(dim='cause').sum(dim='cause')
            df_total = pd.DataFrame(
                total_dalys.values,
                index=self.human_health.lat.values,
                columns=self.human_health.lon.values
            )
            df_total.to_excel(writer, sheet_name="Sum_DALY_wo_turb")

            # --- Sheets 5+: DALYs per impact with turbines ---
            for varname in self.human_health.data_vars:
                data = self.human_health[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_w_turb")

            # --- Final Sheet: Total DALYs (sum of all layers) ---
            total_dalys = self.human_health.to_array(dim='cause').sum(dim='cause')
            df_total = pd.DataFrame(
                total_dalys.values,
                index=self.human_health.lat.values,
                columns=self.human_health.lon.values
            )
            df_total.to_excel(writer, sheet_name="Sum_DALY_w_turb")

            # Same, with normalized per kWh values
            for varname in self.human_health_per_kWh_wo_turbines:
                data = self.human_health_per_kWh_wo_turbines[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_wo_turb_kWh")

            # Same, with normalized per kWh values
            for varname in self.human_health_per_kWh:
                data = self.human_health_per_kWh[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_w_turb_kWh")

            # --- Final Sheet: Total DALYs (sum of all layers) ---
            total_dalys = self.human_health_per_kWh_wo_turbines.to_array(dim='cause').sum(dim='cause')
            df_total = pd.DataFrame(total_dalys.values, index=self.human_health.lat.values,
                                    columns=self.human_health.lon.values)
            df_total.to_excel(writer, sheet_name="Total_DALYs_wo_turbines_per_kWh")

            total_dalys = self.human_health_per_kWh.to_array(dim='cause').sum(dim='cause')
            df_total = pd.DataFrame(total_dalys.values, index=self.human_health.lat.values,
                                    columns=self.human_health.lon.values)
            df_total.to_excel(writer, sheet_name="Sum_DALY_w_turb_kWh")

            # calculate delta DALYs
            delta_dalys = self.human_health - self.human_health_wo_turbines
            # replace negative values with zeros
            delta_dalys = xr.where(delta_dalys < 0, 0, delta_dalys)
            delta_dalys_per_kWh = self.human_health_per_kWh - self.human_health_per_kWh_wo_turbines
            # replace negative values with zeros
            delta_dalys_per_kWh = xr.where(delta_dalys_per_kWh < 0, 0, delta_dalys_per_kWh)

            # --- Sheet: Delta DALYs (with turbines - without turbines) ---
            for varname in delta_dalys.data_vars:
                data = delta_dalys[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_diff")

            # --- Final Sheet: Total Delta DALYs (sum of all layers) ---
            total_delta_dalys = delta_dalys.to_array(dim='cause').sum(dim='cause')
            df_total_delta = pd.DataFrame(
                total_delta_dalys.values,
                index=self.human_health.lat.values,
                columns=self.human_health.lon.values
            )
            df_total_delta.to_excel(writer, sheet_name="Sum_DALY_diff")

            # --- Sheet: Delta DALYs per kWh (with turbines - without turbines) ---
            for varname in delta_dalys_per_kWh.data_vars:
                data = delta_dalys_per_kWh[varname].values
                df = pd.DataFrame(data, index=self.human_health.lat.values, columns=self.human_health.lon.values)
                # Sheet names must be ≤ 31 chars
                safe_name = varname[:31]
                df.to_excel(writer, sheet_name=f"{safe_name}_diff_kWh")

            # --- Final Sheet: Total Delta DALYs per kWh (sum of all layers) ---
            total_delta_dalys_per_kWh = delta_dalys_per_kWh.to_array(dim='cause').sum(dim='cause')
            df_total_delta_per_kWh = pd.DataFrame(
                total_delta_dalys_per_kWh.values,
                index=self.human_health.lat.values,
                columns=self.human_health.lon.values
            )
            df_total_delta_per_kWh.to_excel(writer, sheet_name="Sum_DALY_diff_kWh")




