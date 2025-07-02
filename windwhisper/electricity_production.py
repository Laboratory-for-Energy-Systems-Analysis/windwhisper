import xarray as xr

from . import DATA_DIR

def get_capacity_factor(lat, lon):
    """
    Fetch capacity factor data for a given latitude and longitude.

    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        xarray.DataArray: Capacity factor at the specified location.
    """
    filepath = DATA_DIR / "gwa3_capacityfactor_latlon.nc"
    ds = xr.open_dataset(filepath)

    # Select the data for the given latitude and longitude
    cf = ds.interp(latitude=lat, longitude=lon, method="nearest")["capacity_factor"].values.item(0)
    return cf

def get_electricity_production(lat: float, lon: float, power: int, lifetime: int):
    """
    Fetch electricity production data for a given latitude and longitude.

    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param power: Rated power of the wind turbine in kW.
    :param lifetime: Lifetime of the wind turbine in years.
    :return: Electricity production in kWh over the lifetime.
    """
    capacity_factor = get_capacity_factor(lat, lon)
    down_time = 0.05  # Example downtime percentage

    annual_production = power * capacity_factor * (1 - down_time) * 8760  # kWh/year
    total_production = annual_production * lifetime  # kWh over the lifetime
    return total_production
