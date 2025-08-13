import xarray as xr
import os
import numpy as np

# Directory where you saved the TIFFs
tif_dir = "outputs"

# Order matters to preserve layer names
layers = [
    "road_lden", "road_lnight",
    "rail_lden", "rail_lnight",
    "air_lden", "air_lnight",
    "ind_lden", "ind_lnight",
]

# Read each layer into a DataArray
dataarrays = []
for layer in layers:
    tif_path = os.path.join(tif_dir, f"{layer}_germany.tif")
    da = xr.open_dataarray(tif_path, engine="rasterio")
    da.name = layer
    dataarrays.append(da)

# Merge into a single Dataset
dataset = xr.merge(dataarrays)

for var in dataset.data_vars:
    dataset[var] = dataset[var].astype(np.float32)
dataset.to_netcdf("germany_noise_dataset.nc")
