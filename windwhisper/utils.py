from pyproj import Transformer
import json

from . import HOME_DIR

def create_bounding_box(center_x: float, center_y: float, buffer_meters: int) -> tuple:
    """Create a square bounding box around a point.

    :param center_x: X coordinate of the centre.
    :type center_x: float
    :param center_y: Y coordinate of the centre.
    :type center_y: float
    :param buffer_meters: Distance applied in all directions.
    :type buffer_meters: float
    :returns: Bounding box ``(min_x, min_y, max_x, max_y)``.
    :rtype: tuple[float, float, float, float]
    """
    min_x = center_x - buffer_meters
    max_x = center_x + buffer_meters
    min_y = center_y - buffer_meters
    max_y = center_y + buffer_meters

    return min_x, min_y, max_x, max_y


def translate_4326_to_3035(lon: float, lat: float) -> tuple:
    """Project geographic coordinates to the European LAEA CRS.

    :param lon: Longitude in degrees.
    :type lon: float
    :param lat: Latitude in degrees.
    :type lat: float
    :returns: Projected ``(x, y)`` coordinates in metres.
    :rtype: tuple[float, float]
    """

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    return transformer.transform(lon, lat)


def load_secret():
    """Load the Google API key from the project secret file.

    :returns: API key string when available, otherwise ``None``.
    :rtype: str | None
    """

    try:
        with open(f"{HOME_DIR}/secret.json") as f:
            data = json.load(f)
            return data["google_api_key"]
    except Exception:
        return None
