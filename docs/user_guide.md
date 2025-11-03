# User guide

This guide summarises the main workflow stages described in the project README
and links them to the modules that implement each step.

## 1. Prepare turbine inputs

Use :mod:`windwhisper.windturbines` to validate the input catalogue and enrich
it with predicted sound power levels. The :class:`~windwhisper.windturbines.TurbineModel`
class bundles training and inference helpers for the machine learning model that
estimates broadband noise.

## 2. Load wind resource data

The :mod:`windwhisper.windspeed` module can ingest packaged ERA5 climatology or
fetch NEWA data through the API. It exposes utilities to interpolate the hourly
wind profiles at hub height for the geographic positions of the turbines.

## 3. Simulate propagation

The propagation stack is implemented in :mod:`windwhisper.noisepropagation` and
:mod:`windwhisper.noiseanalysis`. These modules define the analysis grid, compute
attenuation layers, and aggregate the turbine emissions into $L_{den}$ and
$L_{night}$ metrics.

## 4. Combine with ambient noise

Retrieve contextual rasters through :mod:`windwhisper.ambient_noise` and merge
them with the simulated emissions. The convenience functions in
:mod:`windwhisper.noiseanalysis` expose methods to create composite maps and to
export GeoJSON contours.

## 5. Evaluate health impacts

The :mod:`windwhisper.health_impacts` package brings together population grids,
health statistics, and acoustic outputs. Use the :class:`~windwhisper.health_impacts.HumanHealth`
class to compute disability-adjusted life years (DALY) and energy-normalised
indicators.

## 6. Visualise and export

Finally, :mod:`windwhisper.plotting` and :mod:`windwhisper.electricity_production`
contain helpers to render interactive folium maps, export HTML reports, and
summarise energy production results alongside the acoustic findings.

Refer to the example notebook in ``dev/human health.ipynb`` for an end-to-end
analysis and the README for a detailed walkthrough with required inputs.
