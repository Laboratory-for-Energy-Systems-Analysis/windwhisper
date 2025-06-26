
import math
import json

# Load health_parameters.json from data
with open("data/health_parameters.json", "r") as f:
    params = json.load(f)

import pandas as pd

# Load data from data_diseases_europe and Population_data
disease_df = pd.read_csv("data/data_diseases_europe.csv", sep = ";")
population_df = pd.read_csv("data/Population_data.csv", sep = ";")

# unify column names
disease_df.columns = disease_df.columns.str.lower()
population_df.columns = population_df.columns.str.lower()


def get_disease_totals(country, disease, disease_df):
    row = disease_df[
        (disease_df['country'].str.lower() == country.lower()) &
        (disease_df['disease'].str.lower() == disease.lower())
    ]
    if row.empty:
        raise ValueError(f"No data found for {disease} in {country}")
    return float(row['yld'].values[0]), float(row['yll'].values[0])

def get_population_totals(country, population_df):
    row = population_df[
        (population_df['country'].str.lower() == country.lower())
    ]
    if row.empty:
        raise ValueError(f"No data found for population in {country}")
    return float(row['population'].values[0])

# 1. Highly Annoyed
def calculate_highly_annoyed_dalys(exposed_individuals, lden, lifetime, noise_type_ha):
    p = params['highly_annoyed'][noise_type_ha]
    a = p['a']
    b = p['b']
    c = p['c']
    disability_weight = p['disability_weight']
    percentage_affected = a + b * lden + c * (lden ** 2)
    affected = exposed_individuals * (percentage_affected / 100)
    yld = affected * disability_weight
    dalys_per_year = yld
    dalys = dalys_per_year * lifetime
    return dalys

# 2. High Sleep Disorder
def calculate_high_sleep_disorder_dalys(exposed_individuals, lnight, lifetime, noise_type_hsd):
    p = params['high_sleep_disorder'][noise_type_hsd]
    a = p['a']
    b = p['b']
    c = p['c']
    disability_weight = p['disability_weight']
    percentage_affected = a + b * lnight + c * (lnight ** 2)
    affected = exposed_individuals * (percentage_affected / 100)
    yld = affected * disability_weight
    dalys_per_year = yld
    dalys = dalys_per_year * lifetime
    return dalys

# 3. Ischemic Heart Disease (IHD)
def calculate_ihd_dalys(exposed_individuals, lden, lifetime, noise_type, country):
    p = params['ischemic_heart_disease'][noise_type]
    rr_per_10db = p['a']
    threshold = p['threshold']
    if lden < threshold:
        return 0.0
    population = get_population_totals(country, population_df)
    population_rate = exposed_individuals / population
    rr = math.exp((math.log(rr_per_10db) / 10) * (lden - threshold))
    paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
    yld_total, yll_total = get_disease_totals(country, "ischemic_heart_disease", disease_df)
    yld_noise = yld_total * paf
    yll_noise = yll_total * paf
    dalys_per_year = yld_noise + yll_noise
    dalys = dalys_per_year * lifetime
    return dalys

# 4. Diabetes
def calculate_diabetes_dalys(exposed_individuals, lden, lifetime, noise_type, country):
    p = params['diabetes'][noise_type]
    rr_per_10db = p['a']
    threshold = p['threshold']
    if lden < threshold:
        return 0.0
    population = get_population_totals(country, population_df)
    population_rate = exposed_individuals / population
    rr = math.exp((math.log(rr_per_10db) / 10) * (lden - threshold))
    paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
    yld_total, yll_total = get_disease_totals(country, "diabetes", disease_df)
    yld_noise = yld_total * paf
    yll_noise = yll_total * paf
    dalys_per_year = yld_noise + yll_noise
    dalys = dalys_per_year * lifetime
    return dalys
# 5. Stroke
def calculate_stroke_dalys(exposed_individuals, lden, lifetime, noise_type, country):
    p = params['stroke'][noise_type]
    rr_per_10db = p['a']
    threshold = p['threshold']
    if lden < threshold:
        return 0.0
    population = get_population_totals(country, population_df)
    population_rate = exposed_individuals / population
    rr = math.exp((math.log(rr_per_10db) / 10) * (lden - threshold))
    paf = population_rate * (rr - 1) / (population_rate * (rr - 1) + 1)
    yld_total, yll_total = get_disease_totals(country, "stroke", disease_df)
    yld_noise = yld_total * paf
    yll_noise = yll_total * paf
    dalys_per_year = yld_noise + yll_noise
    dalys = dalys_per_year * lifetime
    return dalys

def calculate_total_dalys(exposed_individuals, lden, lnight, lifetime,  noise_type_ha, noise_type_hsd, noise_type, country):
    dalys_ha = calculate_highly_annoyed_dalys(exposed_individuals, lden, lifetime, noise_type_ha)
    dalys_hsd = calculate_high_sleep_disorder_dalys(exposed_individuals, lnight, lifetime, noise_type_hsd)
    dalys_ihd = calculate_ihd_dalys(exposed_individuals, lden, lifetime, noise_type, country)
    dalys_diabetes = calculate_diabetes_dalys(exposed_individuals, lden, lifetime, noise_type, country)
    dalys_stroke = calculate_stroke_dalys(exposed_individuals, lden, lifetime, noise_type, country)
    total_dalys = dalys_ha + dalys_hsd + dalys_ihd + dalys_diabetes + dalys_stroke

    return total_dalys

if __name__ == "__main__":
    country = ("European Region")
    lden = 50
    lnight = 50
    noise_type_ha = "road_without_alpinestudies"
    noise_type_hsd = "combined"
    noise_type = "road middle"
    exposed = 20000
    lifetime = 20

    print("Country:", country)
    print("Noise Level (Lden, Lnight):", lden,"dB(A)",",", lnight,"dB(A)",)
    print("Exposed Population:", exposed, "people")
    print("Lifetime of the wind turbine:", lifetime, "years")
    print("Highly Annoyed:", calculate_highly_annoyed_dalys(exposed, lden, lifetime, noise_type_ha),"DALYs")
    print("Sleep Disorder:", calculate_high_sleep_disorder_dalys(exposed, lnight, lifetime, noise_type_hsd),"DALYs")
    print("IHD:", calculate_ihd_dalys(exposed, lden, lifetime, noise_type, country),"DALYs")
    print("Diabetes:", calculate_diabetes_dalys(exposed, lden, lifetime, noise_type, country),"DALYs")
    print("Stroke:", calculate_stroke_dalys(exposed, lden, lifetime, noise_type, country),"DALYs")
    print("Total:", calculate_total_dalys(exposed, lden, lnight, lifetime, noise_type_ha, noise_type_hsd, noise_type, country),"DALYs")

