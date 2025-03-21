from windwhisper import WindTurbines, NoisePropagation, NoiseAnalysis
import xarray as xr

# time to test the new workflow
import time


# we can preload the wind speed data, otherwise, the tool will do it every time
filepath_wind_speed = "/Users/romain/GitHub/windwhisper/dev/fixtures/era5_mean_2013-2022_month_by_hour.nc"
filepath_correction = "/Users/romain/GitHub/windwhisper/dev/fixtures/ratio_gwa2_era5.nc"

def wind_speed_data():

    wind_speed = xr.open_dataset(filepath_wind_speed).to_array().mean(dim="month")
    correction = xr.open_dataset(filepath_correction).to_array()
    correction = correction.sel(variable='ratio_gwa2_era5_mean_WS').interp(latitude=wind_speed.latitude, longitude=wind_speed.longitude, method="linear")
    return wind_speed * correction

wind_speed_data = wind_speed_data()

wind_turbines = {
    'Turbine 0':
     {
        'diameter': 70.0,
        'hub height': 85.0,
        'position': (43.45111343125036, 5.2518247370645215),
        'power': 2500.0
     },
}

t = time.process_time()

wt = WindTurbines(
    wind_turbines=wind_turbines,
    wind_speed_data=wind_speed_data,
    #retrain_model=True
)

elevation_data = xr.open_dataset("fixtures/Copernicus_DSM_90m_COG.nc")

noise_prop = NoisePropagation(
    wind_turbines=wt.wind_turbines,
    humidity=70,
    temperature=20,
    elevation_data=elevation_data,
)

noise_analysis = NoiseAnalysis(
    noise_propagation=noise_prop,
    wind_turbines=wt.wind_turbines,
)

noise_analysis.generate_map(
    filepath="my_noise_map.html"
)

elapsed_time = time.process_time() - t
print(elapsed_time)