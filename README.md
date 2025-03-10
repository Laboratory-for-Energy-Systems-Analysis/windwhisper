# windwhisper

``windwhisper`` is a Python package for estimating wind turbine 
noise propagation and its impacts on surrounding populations.

## Installation

As ``windwhisper`` is being actively developed, 
it is best to install from Github using ``pip``:

```bash
  pip install git+https://github.com/Laboratory-for-Energy-Systems-Analysis/windwhisper.git
```

## Usage

``windwhisper`` can be used to estimate the noise propagation from wind turbines. 
The following example shows how to estimate the noise propagation from a series of
wind turbines, and the exposure of a number of listeners to noise, expressed using
the L_den indicator. Results can be exported on a map.


### Initializing wind turbines and listeners:

```python

    from windwhisper import WindTurbines, NoisePropagation, NoiseAnalysis
    import xarray as xr
    
    # we can preload the wind speed data, otherwise, the tool will do it every time
    # if you do not have wind data, the tool will fetch data from the New Wind Atlas API instead
    
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
    
    wt = WindTurbines(
        wind_turbines=wind_turbines,
        wind_speed_data=wind_speed_data,
        #retrain_model=True
    )
    
    # Noise propagation calculation
    # you can provide local elevation data
    # if not, the tool will fetch data from the Google Maps API 
    # (you need to provide an API token in a secret.json file in teh root folder)
    elevation_data = xr.open_dataset("fixtures/Copernicus_DSM_90m_COG.nc")

    noise_prop = NoisePropagation(
        wind_turbines=wt.wind_turbines,
        humidity=70,
        temperature=20,
        elevation_data=elevation_data,
    )
    
    # Noise analysis
    noise_analysis = NoiseAnalysis(
        noise_propagation=noise_prop,
        wind_turbines=wt.wind_turbines,
    )


```


### Noise Map Generation

An HTML map can exported, with noise contours for different L_den noise levels (in dB(A)).

```python

    noise_analysis.generate_map(
        filepath="my_noise_map.html"
    )

```

### GeoJSON export

Alternatively, GeoJSON objects from contours can be produced from the raster data.
`.get_geojson_contours()? fetches geoJson L_den noise contour objects for:

* ambient noise sources (from the EU Noise maps, including industry, road, rail, and air traffic)
* noise from the wind turbines, 
* combined ambient and wind turbines' noise
* net wind turbines noise contribution
* and noise levels above EU guidelines (i.e., 55 dB(A)) as a result of the wind turbines implementation.


```python
    
    noise_analysis.get_geojson_contours()

```

## License

``windwhisper`` is distributed under the terms of the BSD 3-Clause license (see LICENSE).

## Authors

* Romain Sacchi (romain.sacchi@psi.ch), Paul Scherrer Institut (PSI)
* Maxime Balandret, Paul Scherrer Institut (PSI)

## Acknowledgements
The development of `windwhisper` is supported by the European project
[WIMBY](https://cordis.europa.eu/project/id/101083460) (Wind In My BackYard, grant agreement No 101083460).
