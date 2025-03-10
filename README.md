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

    from windwhisper import windturbines
    import xarray as xr
    
    # we can preload the wind speed data, otherwise, the tool will do it every time
    filepath_wind_speed = "/Users/romain/GitHub/windwhisper/dev/fixtures/era5_mean_2013-2022_month_by_hour.nc"
    filepath_correction = "/Users/romain/GitHub/windwhisper/dev/fixtures/ratio_gwa2_era5.nc"
    
    def wind_speed_data():
        wind_speed = xr.open_dataset(filepath_wind_speed).to_array().mean(dim="month")
        correction = xr.open_dataset(filepath_correction).to_array()
        correction = correction.sel(variable='ratio_gwa2_era5_mean_WS').interp(latitude=wind_speed.latitude, longitude=wind_speed.longitude, method="linear")
        return wind_speed * correction
        
    data = wind_speed_data()
    
    # list of wind turbines to analyze
    wind_turbines = {
        'Turbine 0': 
         {
             'diameter': 70.0,
            'hub height': 85.0,
            'position': (43.45111343125036, 5.2518247370645215),
            'power': 2500.0
         },
    }
    
    wt = windturbines.WindTurbines(
        wind_turbines=wind_turbines,
        wind_speed_data=data,
        #retrain_model=True # if you want to retrain the model or train it for the first time.
    )

```


### Noise Map Generation

An HTML map is exported, with noise contours for different L_den noise levels (in dB(A)).

```python
    wt.noise_analysis.generate_map()
```

### GeoJSON export

Alternatively, GeoJSON objects from contours can be produced from the raster data.

```python
    import matplotlib.pyplot as plt
    import geojson
    from shapely.geometry import LineString, mapping
    
    color_map = {
        30: "green",
        40: "yellow",
        55: "orange",
        60: "red",
        70: "purple",
    }
    levels=[30, 40, 55, 60]
    
    def create_geojson(data):
        # Generate the contours
        c = plt.contour(
            data.lon, data.lat, data.data,
            levels=levels,
            colors=[color_map[k] for k in color_map.keys()],
            linewidths=1.5
        )
    
        # Create a list to store GeoJSON features
        geojson_features = []
    
        for i, collection in enumerate(c.collections):
            for path in collection.get_paths():
                for line in path.to_polygons():
                    coords = [
                        (x, y) for x, y in line
                    ]
    
                    # Add the geometry as a GeoJSON feature
                    geojson_features.append(
                        geojson.Feature(
                            geometry=mapping(LineString(coords)),
                            properties={
                                "level": c.levels[i],
                                "color": color_map[c.levels[i]]
                            }
                        )
                    )
    
        # Create a GeoJSON FeatureCollection
        return geojson.FeatureCollection(geojson_features)
    
    geojson_objs = []
    
    wt = windturbines.WindTurbines(
        wind_turbines=wind_turbines, # some list of dictionaries containing wind turbines locations, power, etc.
        wind_speed_data=data,
    )

    geojson_obj = create_geojson(wt.noise_analysis.merged_map["net"]) # for net noise contribution. Other attributes are available: "wind", "ambient", and "combined".

```

## License

``windwhisper`` is distributed under the terms of the BSD 3-Clause license (see LICENSE).

## Authors

* Romain Sacchi (romain.sacchi@psi.ch), Paul Scherrer Institut (PSI)
* Maxime Balandret, Paul Scherrer Institut (PSI)

## Acknowledgements
The development of `windwhisper` is supported by the European project
[WIMBY](https://cordis.europa.eu/project/id/101083460) (Wind In My BackYard, grant agreement No 101083460).
