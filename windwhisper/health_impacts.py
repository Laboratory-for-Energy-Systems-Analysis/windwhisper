import math
import json
import pandas as pd
import numpy as np
import xarray as xr


def load_human_health_parameters() -> dict:
    """
    Load human health parameters from a JSON file.
    """
    with open("data/human_health_parameters.json", "r") as f:
        return json.load(f)

def load_disease_data() -> pd.DataFrame:
    return pd.read_csv("data/data_diseases_europe.csv", sep=";")

def load_population_data() -> pd.DataFrame:
    return pd.read_csv("data/Population_data.csv", sep=";")



def get_population_totals(country: str) -> float:
    """
    Returns the total population for a given country.
    :return: Total population for the specified country.
    :raises ValueError: If no population data is found for the specified country.
    """
    population = load_population_data()
    row = population.loc[(population["country"].str.lower() == country)]

    if row.empty:
        raise ValueError(f"No data found for population in {country}")
    return float(row["population"].values[0])


class HumanHealth():
    def __init__(self, country: str, l_den: xr.DataArray , l_night: xr.DataArray, lifetime : int = 20):
        self.country = country.lower()
        self.l_den = l_den
        self.l_night = l_night
        self.lifetime = lifetime
        self.disease_data = load_disease_data()
        self.population = get_population_totals()
        self.human_health_parameters = load_human_health_parameters()


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
            (self.disease_data["country"].str.lower() == self.country)
            & (self.disease_data["disease"].str.lower() == disease.lower())
        ]
        if row.empty:
            raise ValueError(f"No data found for {disease} in {self.country}")
        return float(row["yld"].values[0]), float(row["yll"].values[0])



    def calculate_highly_annoyed_dalys(
        self,
        exposed_individuals: np.ndarray,
        noise_type_ha: str
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for highly annoyed individuals
        based on the noise level (Lden) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
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
        affected = exposed_individuals * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime
        return dalys


    # 2. High Sleep Disorder
    def calculate_high_sleep_disorder_dalys(
        self,
        exposed_individuals: np.ndarray,
        noise_type_hsd: str
    )-> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for high sleep disorder
        based on the noise level (Lnight) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
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
        affected = exposed_individuals * (percentage_affected / 100)
        yld = affected * disability_weight
        dalys_per_year = yld
        dalys = dalys_per_year * self.lifetime
        return dalys


    # 3. Ischemic Heart Disease (IHD)
    def calculate_ihd_dalys(
        self,
        exposed_individuals: np.ndarray,
        noise_type: str,
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for ischemic heart disease
        based on the noise level (Lden) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for ischemic heart disease.
        """
        p = self.human_health_parameters["ischemic_heart_disease"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        if self.l_den < threshold:
            return 0.0

        population_rate = exposed_individuals / self.population
        rr = math.exp((math.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("ischemic_heart_disease")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime

        return dalys


    # 4. Diabetes
    def calculate_diabetes_dalys(
        self,
        exposed_individuals: np.ndarray,
        noise_type: str,
    ) -> np.ndarray:
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for diabetes
        based on the noise level (Lden) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for diabetes.
        """
        p = self.human_health_parameters["diabetes"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        if self.l_den < threshold:
            return 0.0

        population_rate = exposed_individuals / self.population
        rr = math.exp((math.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("diabetes")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime
        return dalys


    # 5. Stroke
    def calculate_stroke_dalys(
            self,
            exposed_individuals: np.ndarray,
            noise_type: str,
    ):
        """
        Calculate the Disability-Adjusted Life Years (DALYs) for stroke
        based on the noise level (Lden) and the number of exposed individuals.
        :param exposed_individuals: Number of individuals exposed to noise.
        :param lden: Noise level in Lden (day-evening-night noise level).
        :param noise_type: Type of noise (e.g., "road middle", "railway").
        :return: DALYs for stroke.
        """
        p = self.human_health_parameters["stroke"][noise_type]
        rr_per_10db = p["a"]
        threshold = p["threshold"]

        if self.l_den < threshold:
            return 0.0

        population_rate = exposed_individuals / self.population
        rr = math.exp((math.log(rr_per_10db) / 10) * (self.l_den - threshold))
        paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
        yld_total, yll_total = self.get_disease_totals("stroke")
        yld_noise = yld_total * paf
        yll_noise = yll_total * paf
        dalys_per_year = yld_noise + yll_noise
        dalys = dalys_per_year * self.lifetime
        return dalys


    def calculate_total_dalys(
        self,
        exposed_individuals: np.ndarray,
        noise_type_ha: str,
        noise_type_hsd: str,
        noise_type: str,
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
        dalys_ha = self.calculate_highly_annoyed_dalys(
            exposed_individuals, noise_type_ha
        )
        dalys_hsd = self.calculate_high_sleep_disorder_dalys(
            exposed_individuals, noise_type_hsd
        )
        dalys_ihd = self.calculate_ihd_dalys(
            exposed_individuals, noise_type
        )
        dalys_diabetes = self.calculate_diabetes_dalys(
            exposed_individuals, noise_type
        )
        dalys_stroke = self.calculate_stroke_dalys(
            exposed_individuals, noise_type
        )
        total_dalys = dalys_ha + dalys_hsd + dalys_ihd + dalys_diabetes + dalys_stroke

        return total_dalys


# if __name__ == "__main__":
#     country = "European Region"
#     lden = 50
#     lnight = 50
#     noise_type_ha = "road_without_alpinestudies"
#     noise_type_hsd = "combined"
#     noise_type = "road middle"
#     exposed = 20000
#     lifetime = 20
