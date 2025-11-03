import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import Map, Element
import geojson
import numpy as np
from shapely.geometry import LineString, mapping, Polygon

color_map = {
        #10: "blue",
        #20: "darkblue",
        30: "green",
        40: "yellow",
        55: "orange",
        60: "red",
        70: "purple",
    }


def generate_contours(data, levels, name="Noise Contours"):
    """Generate contour polygons and wrap them into a Folium layer.

    :param data: Raster to contour.
    :type data: xarray.DataArray
    :param levels: Contour levels in dB.
    :type levels: list[int]
    :param name: Name of the resulting Folium layer.
    :type name: str
    :returns: Folium feature group containing contour polylines.
    :rtype: folium.FeatureGroup
    """

    # Generate the contours
    c = plt.contour(
        data.lon, data.lat, data.data,
        levels=levels,
        colors=[color_map[k] for k in color_map.keys()],
        linewidths=1.5
    )

    # Create a list to store GeoJSON features
    geojson_features = []

    for i, path in enumerate(c.get_paths()):
        for line in path.to_polygons():
            coords = [
                (x, y) for x, y in line
            ]

            # Rotation angle in degrees
            rotation_angle = 0  # Adjust as needed

            # Create a Shapely Polygon object
            polygon = Polygon(coords)

            # Calculate the centroid
            centroid = polygon.centroid
            centroid_coords = (centroid.y, centroid.x)  # Latitude, Longitude

            # Apply rotation
            rotated_polygon_coords = rotate_coordinates(coords, rotation_angle, centroid_coords)

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
    geojson_object = geojson.FeatureCollection(geojson_features)

    # Add GeoJSON to a Folium FeatureGroup
    feature_group = folium.FeatureGroup(name=name)
    folium.GeoJson(
        geojson_object,
        style_function=lambda feature: {
            "color": feature["properties"]["color"],
            "weight": 2,
        }
    ).add_to(feature_group)

    return feature_group


def add_legend(map_object):
    """Append a categorical legend to a Folium map.

    :param map_object: Folium map receiving the legend.
    :type map_object: folium.Map
    :returns: The modified map object.
    :rtype: folium.Map
    """
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 15px; width: 200px; height: auto; 
        background-color: white; z-index:9999; font-size:14px; 
        border:2px solid grey; padding: 10px; border-radius: 5px;
    ">
    <strong>Legend</strong><br>
    """
    for level, color in color_map.items():
        legend_html += f"""
        <i style="background: {color}; width: 10px; height: 10px; display: inline-block; margin-right: 10px;"></i>
        {level} dB(A)<br>
        """
    legend_html += "</div>"

    # Wrap the HTML in a folium Element and add it to the map
    legend = Element(legend_html)
    map_object.get_root().html.add_child(legend)

    return map_object

def rotate_coordinates(coords, angle, center):
    """Rotate coordinates around a centroid.

    :param coords: Sequence of ``(latitude, longitude)`` pairs.
    :type coords: list[tuple[float, float]]
    :param angle: Rotation angle in degrees.
    :type angle: float
    :param center: Rotation centre expressed as ``(latitude, longitude)``.
    :type center: tuple[float, float]
    :returns: Rotated coordinates as ``(latitude, longitude)`` pairs.
    :rtype: list[tuple[float, float]]
    """
    angle_rad = np.radians(angle)
    center_x, center_y = center

    rotated_coords = []
    for lat, lon in coords:
        # Translate point to origin
        translated_x = lon - center_x
        translated_y = lat - center_y

        # Apply rotation matrix
        rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
        rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)

        # Translate point back
        final_x = rotated_x + center_x
        final_y = rotated_y + center_y
        rotated_coords.append((final_y, final_x))  # Latitude, Longitude

    return rotated_coords

def generate_map(
        noise_dataset,
        filepath="noise_map.html"
):
    """Export contour overlays for the provided noise dataset.

    :param noise_dataset: Dataset containing combined noise layers.
    :type noise_dataset: xarray.Dataset
    :param filepath: Destination filepath for the generated HTML map.
    :type filepath: str
    :returns: Generated Folium map instance.
    :rtype: folium.Map
    """

    # Create Folium map
    center_lat = np.mean(noise_dataset.lat.values)
    center_lon = np.mean(noise_dataset.lon.values)

    m = folium.Map(location=[center_lat, center_lon])

    if "combined" in noise_dataset:
        # Add the contours to the map
        contours = generate_contours(
            noise_dataset["combined"],
            levels=[30, 40, 55, 60, 70],
            name="Overall noise levels"
        )
        m.add_child(contours)

    if "wind" in noise_dataset:
        contours = generate_contours(
            noise_dataset["wind"],
            levels=[
                # 20,
                30,
                40,
                55,
                60,
                70
            ],
            name="Wind turbine(s) noise levels"
        )
        m.add_child(contours)

    if "ambient" in noise_dataset:
        contours = generate_contours(
            noise_dataset["ambient"],
            levels=[30, 40, 55, 60, 70],
            name="Pre-existing noise levels"
        )
        m.add_child(contours)

    if "net" in noise_dataset:
        contours = generate_contours(
            noise_dataset["net"],
            levels=[30, 40, 55, 60],
            name="Net contribution"
        )
        m.add_child(contours)

    if "flip" in noise_dataset:
        contours = generate_contours(
            noise_dataset["flip"],
            levels=[55, 60, 70],
            name="Area beyond EU guidelines (55 db(A))"
        )
        m.add_child(contours)

    # Add the legend
    m = add_legend(m)

    # Add layer control for toggling
    folium.LayerControl().add_to(m)

    # Save the map as an HTML file
    m.save(filepath)
    return m


def create_geojson(data):
    """Generate a GeoJSON feature collection of contour polylines.

    :param data: Raster to contour.
    :type data: xarray.DataArray
    :returns: GeoJSON feature collection describing the contour lines.
    :rtype: geojson.FeatureCollection
    """

    color_map = {
        30: "green",
        40: "yellow",
        55: "orange",
        60: "red",
        70: "purple",
    }
    levels = [30, 40, 55, 60]

    # Generate the contours
    c = plt.contour(
        data.lon, data.lat, data.data,
        levels=levels,
        colors=[color_map[k] for k in color_map.keys()],
        linewidths=1.5
    )

    # Create a list to store GeoJSON features
    geojson_features = []

    for i, segs in enumerate(c.allsegs):
        for line in segs:
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
