# Overview

WindWhisper models the acoustic footprint of onshore wind turbine projects. The
package bundles utilities to ingest turbine configurations, compute wind speed
profiles, propagate the resulting sound power levels, and contextualise the
findings with ambient noise measurements and health impact metrics.

## Feature highlights

- **Wind resource modelling** – load ERA5 or NEWA data, apply correction factors,
  and interpolate hub-height wind speeds for project locations.
- **Noise propagation** – compute distance, atmospheric absorption, ground
  effects, and obstacle attenuation to estimate $L_{den}$ and $L_{night}$
  values across a study grid.
- **Ambient context** – merge local emissions with EU noise rasters to determine
  whether wind projects increase community sound exposure.
- **Health assessments** – combine the acoustic fields with demographic datasets
  to estimate disability-adjusted life years (DALY) and other indicators for
  affected populations.

WindWhisper is maintained by the Laboratory for Energy Systems Analysis at the
Paul Scherrer Institute. The source code and issue tracker are available on
[GitHub](https://github.com/Laboratory-for-Energy-Systems-Analysis/windwhisper).
