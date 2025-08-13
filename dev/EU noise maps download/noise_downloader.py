import os
import time
import argparse
import numpy as np
import xarray as xr
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PIXEL_VALUE_TO_LDEN = {v: v for v in range(256)}  # Replace with actual mapping

NOISE_URLS = {
    "road_lden": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_road_lden/ImageServer/exportImage",
    "rail_lden": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_rail_lden/ImageServer/exportImage",
    "air_lden": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_air_lden/ImageServer/exportImage",
    "ind_lden": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_ind_lden/ImageServer/exportImage",
    "road_lnight": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_road_lnight/ImageServer/exportImage",
    "rail_lnight": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_rail_lnight/ImageServer/exportImage",
    "air_lnight": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_air_lnight/ImageServer/exportImage",
    "ind_lnight": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_ind_lnight/ImageServer/exportImage",
}

REGIONS = {
    "germany": (4050000, 4750000, 2800000, 3400000),
    "berlin": (4310000, 4360000, 3060000, 3100000),
}

def safe_filename(x0, y0):
    return f"tile_{x0}_{y0}.tif"

def get_cached_tile_path(cache_dir, x0, y0):
    return os.path.join(cache_dir, safe_filename(x0, y0))

def download_tile_with_cache(url, x0, x1, y0, y1, resolution, cache_dir, retries=3, delay=1.0):
    os.makedirs(cache_dir, exist_ok=True)
    tile_path = get_cached_tile_path(cache_dir, x0, y0)

    if os.path.exists(tile_path):
        try:
            with rasterio.open(tile_path) as src:
                return src.read(1)
        except Exception as e:
            print(f"[!] Failed to open cached tile {tile_path}: {e}")
            return None

    params = {
        "bbox": f"{x0},{y0},{x1},{y1}",
        "bboxSR": "3035",
        "size": f"{resolution[1]},{resolution[0]}",
        "format": "tiff",
        "f": "image"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code != 200:
                print(f"[{x0},{y0}] HTTP error {response.status_code}")
                continue

            content_type = response.headers.get("Content-Type", "")
            if "image/tiff" not in content_type:
                print(f"[{x0},{y0}] Invalid content type: {content_type}")
                print(response.content[:200])
                return None

            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset:
                    data = dataset.read(1)
                    data = np.nan_to_num(data, nan=15)
                    data = np.vectorize(lambda x: PIXEL_VALUE_TO_LDEN.get(x, 0))(data)

                    with rasterio.open(
                        tile_path,
                        "w",
                        driver="GTiff",
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs="EPSG:3035",
                        transform=dataset.transform
                    ) as dst:
                        dst.write(data, 1)

                    return data

        except Exception as e:
            print(f"[{x0},{y0}] Error: {e}")

        time.sleep(delay * (2 ** attempt))

    print(f"[{x0},{y0}] Giving up after {retries} retries")
    return None

def download_single_tile(args):
    return args[1], args[3], download_tile_with_cache(*args)

def fetch_map_parallel(url, xmin, xmax, ymin, ymax, cache_dir, tile_size=10000, pixels=256, max_workers=12):
    n_tiles_x = (xmax - xmin) // tile_size
    n_tiles_y = (ymax - ymin) // tile_size

    tasks = []
    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            x0 = xmin + ix * tile_size
            y0 = ymin + iy * tile_size
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            tasks.append((url, x0, x1, y0, y1, (pixels, pixels), cache_dir))

    print(f"Submitting {len(tasks)} tiles to {max_workers} threads")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {executor.submit(download_single_tile, t): (t[1], t[3]) for t in tasks}
        for future in as_completed(future_to_tile):
            x0, y0 = future_to_tile[future]
            try:
                _, _, data = future.result()
                results[(x0, y0)] = data
            except Exception as e:
                print(f"Error in tile ({x0},{y0}): {e}")
                results[(x0, y0)] = None

    full_array = np.zeros(((n_tiles_y * pixels), (n_tiles_x * pixels)))
    n_missing = 0

    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            x0 = xmin + ix * tile_size
            y0 = ymin + iy * tile_size
            data = results.get((x0, y0))
            if data is not None:
                full_array[
                    (n_tiles_y - iy - 1)*pixels : (n_tiles_y - iy)*pixels,
                    ix*pixels : (ix+1)*pixels
                ] = data
            else:
                print(f"[!] Tile ({ix},{iy}) at {x0},{y0} failed — filling with zeros")
                n_missing += 1

    print(f"[i] Finished with {n_missing} missing tiles out of {n_tiles_x * n_tiles_y}")

    transform = from_origin(xmin, ymax, tile_size / pixels, tile_size / pixels)
    return xr.DataArray(
        full_array,
        dims=["y", "x"],
        coords={
            "x": np.arange(full_array.shape[1]) * transform.a + transform.c,
            "y": np.arange(full_array.shape[0]) * -transform.e + transform.f
        },
        attrs={"crs": "EPSG:3035", "transform": transform}
    )

def save_to_geotiff(xda, filename):
    transform = xda.attrs["transform"]
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=xda.shape[0],
        width=xda.shape[1],
        count=1,
        dtype=xda.dtype,
        crs=xda.attrs["crs"],
        transform=transform,
    ) as dst:
        dst.write(xda.values, 1)

def main():
    parser = argparse.ArgumentParser(description="Batch download EU noise maps for all layers.")
    parser.add_argument("--region", required=True, choices=REGIONS.keys(), help="Region to download (e.g., berlin, germany)")
    parser.add_argument("--outdir", default="outputs", help="Directory to save GeoTIFFs")
    parser.add_argument("--cache", default="cache", help="Cache directory for tiles")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel threads")
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = REGIONS[args.region]
    os.makedirs(args.outdir, exist_ok=True)

    for layer in tqdm(NOISE_URLS.keys(), desc=f"Downloading layers for {args.region}"):
        print(f"\n--- Downloading {layer} for {args.region} ---")
        url = NOISE_URLS[layer]
        cache_dir = os.path.join(args.cache, layer)

        xda = fetch_map_parallel(url, xmin, xmax, ymin, ymax, cache_dir, max_workers=args.workers)

        if np.all(xda.values == 0):
            print("[!] All tile values are zero — check if downloads failed or bounding box is wrong.")
        else:
            print("[✓] Some data successfully retrieved.")

        outpath = os.path.join(args.outdir, f"{layer}_{args.region}.tif")
        save_to_geotiff(xda, outpath)
        print(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
