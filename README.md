# windwhisper

``windwhisper`` is a Python package for estimating wind turbine 
noise propagation and its impacts on surrounding populations.

## Installation

As ``windwhisper`` is being actively developed, 
it is best to install from Github using ``pip``:

```bash
  pip install git+https://github.com/Laboratory-for-Energy-Systems-Analysis/windwhisper.git
```

## Documentation
The documentation for ``windwhisper`` can be found at
https://windwhisper.readthedocs.io/en/latest.

## Usage

``windwhisper`` can be used to estimate the noise propagation from wind turbines and
to assess combined ambient noise levels around a project area. The quick-start example
below walks through the typical workflow and highlights which inputs you need to
provide at each step.

### Package structure at a glance

* ``windturbines`` – validates turbine specifications, trains or loads an acoustic
  emission model, and predicts sound power levels for 3–12 m/s wind speeds.
* ``windspeed`` – loads ERA5-based mean wind speeds, optionally applies correction
  factors, or falls back to the NEWA API when local files are missing.
* ``noisepropagation`` – computes propagation losses (distance, atmospheric,
  ground, obstacles), derives hourly levels, and aggregates to L_den and L_night.
* ``ambient_noise`` – fetches EU environmental noise rasters and combines them with
  modelled turbine levels.
* ``noiseanalysis`` – merges turbine and ambient layers, exports interactive maps,
  and prepares GeoJSON contour data.
* ``health_impacts`` – estimates population exposure, DALYs, and energy outputs for
  health assessments.

Additional utilities cover terrain handling (``ground_attenuation``), atmospheric
absorption, plotting helpers, and settlement/population data preparation.

### Required inputs and configuration

| Step | Mandatory inputs | Optional inputs & configuration |
| ---- | ---------------- | -------------------------------- |
| ``WindTurbines`` | Dictionary of turbines with ``power`` (kW), ``diameter`` (m), ``hub height`` (m), and ``position`` (latitude, longitude). | ``retrain_model``/``dataset_file`` to fit a custom noise model; pre-loaded ``wind_speed_data`` as an ``xarray.DataArray``. |
| ``WindSpeed`` | – (auto-loads packaged ERA5 climatology). | ``wind_speed_data`` array to bypass file/API loading; NEWA API key via ``API_NEWA`` env var for live downloads. |
| ``NoisePropagation`` | Output from ``WindTurbines`` including ``noise_vs_wind_speed`` and ``mean_wind_speed``. | Relative humidity/temperature, local ``elevation_data`` (``xarray.Dataset``). When omitted, terrain data must be retrievable through the configured elevation service (requires API token in ``secret.json`` as referenced in code comments). |
| ``NoiseAnalysis`` | ``NoisePropagation`` instance. | EU Noise API endpoints via environment variables (``API_EU_NOISE_*``) to obtain ambient rasters. |
| Mapping/health | – | Optional: custom output filepath, DALY parameters, Google elevation API token, etc. |

Ensure the relevant API keys are available in your environment (e.g., ``API_NEWA``)
or configuration files before running a simulation.

### End-to-end example

The snippet below configures a single turbine, runs the propagation model,
exports mapping artefacts, and quantifies the human-health impacts over the
project lifetime.

```python
import xarray as xr

from windwhisper import (
    DATA_DIR,
    WindTurbines,
    NoisePropagation,
    NoiseAnalysis,
    HumanHealth,
)


# 1. Describe your turbines (power in kW, geometry in metres, WGS84 coordinates)
wind_turbines = {
    "Demo turbine": {
        "power": 2_500,
        "diameter": 70,
        "hub height": 85,
        "position": (43.4511, 5.2518),
    }
}

# 2. (Optional) load wind-speed climatology once and reuse it
fixtures = DATA_DIR.parent / "dev" / "fixtures"
wind_speed_file = fixtures / "era5_mean_2013-2022_month_by_hour.nc"
correction_file = fixtures / "ratio_gwa2_era5.nc"

if wind_speed_file.exists() and correction_file.exists():
    wind_speed = xr.open_dataset(wind_speed_file).to_array().mean(dim="month")
    correction = xr.open_dataset(correction_file).to_array()
    correction = correction.sel(
        variable="ratio_gwa2_era5_mean_WS"
    ).interp(
        latitude=wind_speed.latitude,
        longitude=wind_speed.longitude,
        method="linear",
    )
    wind_speed = wind_speed * correction
else:
    wind_speed = None  # WindSpeed will fall back to the configured API

wt = WindTurbines(
    wind_turbines=wind_turbines,
    wind_speed_data=wind_speed,
)

# 3. Provide site conditions; use your own elevation raster when available
elevation_path = fixtures / "Copernicus_DSM_90m_COG.nc"
elevation = xr.open_dataset(elevation_path) if elevation_path.exists() else None

noise_prop = NoisePropagation(
    wind_turbines=wt.wind_turbines,
    humidity=70,
    temperature=20,
    elevation_data=elevation,
)

# 4. Merge with ambient sources, create maps, and export contours
noise_analysis = NoiseAnalysis(
    noise_propagation=noise_prop,
    wind_turbines=wt.wind_turbines,
)
noise_analysis.generate_map(filepath="my_noise_map.html")
ambient, wind, combined, net, flip = noise_analysis.get_geojson_contours()

# 5. Quantify health impacts over the turbines' lifetime
human_health = HumanHealth(
    noise_analysis=noise_analysis,
    lifetime=25,  # optional (defaults to 20 years)
)

print(f"Total DALYs: {human_health.human_health:.2e}")
print(f"DALYs per kWh: {human_health.human_health_per_kWh:.2e}")

human_health.export_to_excel("human_health_summary.xlsx")
```

## License

``windwhisper`` is distributed under the terms of the BSD 3-Clause license (see LICENSE).

## Authors

* Romain Sacchi (romain.sacchi@psi.ch), Paul Scherrer Institut (PSI)
* Maxime Balandret, Paul Scherrer Institut (PSI)

## Acknowledgements
The development of `windwhisper` is supported by the European project
[WIMBY](https://cordis.europa.eu/project/id/101083460) (Wind In My BackYard, grant agreement No 101083460).
