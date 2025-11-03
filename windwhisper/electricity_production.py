import xarray as xr

from . import DATA_DIR

def get_capacity_factor(lat, lon):
    """Interpolate the GWA3 capacity factor at a given location.

    :param lat: Latitude in degrees.
    :type lat: float
    :param lon: Longitude in degrees.
    :type lon: float
    :returns: Capacity factor for the requested coordinates.
    :rtype: float
    """
    filepath = DATA_DIR / "gwa3_capacityfactor_latlon.nc"
    ds = xr.open_dataset(filepath)

    # Select the data for the given latitude and longitude
    cf = ds.interp(latitude=lat, longitude=lon, method="nearest")["capacity_factor"].values.item(0)
    return cf

def get_electricity_production(lat: float, lon: float, power: int, lifetime: int):
    """Estimate the lifetime electricity production for a wind turbine.

    :param lat: Latitude in degrees.
    :type lat: float
    :param lon: Longitude in degrees.
    :type lon: float
    :param power: Rated power of the wind turbine in kW.
    :type power: int
    :param lifetime: Operational lifetime in years.
    :type lifetime: int
    :returns: Energy yield in kWh produced over the lifetime.
    :rtype: float
    """
    capacity_factor = get_capacity_factor(lat, lon)
    down_time = 0.05  # Example downtime percentage

    annual_production = power * capacity_factor * (1 - down_time) * 8760  # kWh/year
    total_production = annual_production * lifetime  # kWh over the lifetime
    return total_production
