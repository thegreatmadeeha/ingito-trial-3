#!/usr/bin/env python3
"""Process daily NASA GPM IMERG NetCDF4 files for Ingito regions."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[2]

REGIONS = {
    "feni": {
        "input": ROOT / "ingito/data/gpm/feni",
        "boundary": ROOT / "sar-flood/aoi/feni.geojson",
        "output": ROOT / "ingito/data/gpm/processed/feni_rainfall.json",
    },
    "south_chattogram": {
        "input": ROOT / "ingito/data/gpm/south_chattogram",
        "boundary": ROOT / "sar-flood/aoi/chattogram_south.geojson",
        "output": ROOT / "ingito/data/gpm/processed/south_chattogram_rainfall.json",
    },
}

VARIABLE_CANDIDATES = (
    "precipitation",
    "precipitationCal",
    "precipitationUncal",
    "precipitationCalMean",
)


def find_precipitation_variable(dataset):
    for name in VARIABLE_CANDIDATES:
        if name in dataset.data_vars:
            return name

    available = list(dataset.data_vars)
    raise ValueError(
        f"No recognised precipitation variable. Available: {available}"
    )


def find_coordinate(dataset, candidates):
    for name in candidates:
        if name in dataset.coords:
            return name
        if name in dataset.variables:
            return name
    raise ValueError(f"Could not find coordinate among {candidates}")


def extract_date(path, dataset):
    match = re.search(r"(\d{8})", path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()

    for key in ("time", "time_bnds"):
        if key in dataset:
            values = np.asarray(dataset[key].values).reshape(-1)
            if values.size:
                return str(np.datetime_as_string(values[0], unit="D"))

    raise ValueError(f"Could not determine observation date: {path.name}")


def process_file(path, boundary):
    with xr.open_dataset(path) as dataset:
        variable = find_precipitation_variable(dataset)
        lat_name = find_coordinate(dataset, ("lat", "latitude"))
        lon_name = find_coordinate(dataset, ("lon", "longitude"))

        rainfall = dataset[variable].squeeze(drop=True)

        if lat_name not in rainfall.dims or lon_name not in rainfall.dims:
            raise ValueError(
                f"Unexpected dimensions for {variable}: {rainfall.dims}"
            )

        rainfall = rainfall.transpose(lat_name, lon_name)
        values = np.asarray(rainfall.values, dtype=np.float64)

        lats = np.asarray(dataset[lat_name].values)
        lons = np.asarray(dataset[lon_name].values)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Use the regional geometry to select pixels.
        inside = np.zeros(lat_grid.shape, dtype=bool)
        for geometry in boundary.geometry:
            if geometry is None or geometry.is_empty:
                continue

            minx, miny, maxx, maxy = geometry.bounds
            candidates = (
                (lon_grid >= minx)
                & (lon_grid <= maxx)
                & (lat_grid >= miny)
                & (lat_grid <= maxy)
            )

            from shapely import contains_xy
            inside |= candidates & contains_xy(
                geometry, lon_grid, lat_grid
            )

        selected = values[inside]
        valid = selected[np.isfinite(selected) & (selected >= 0)]

        if valid.size == 0:
            raise ValueError(f"No valid rainfall pixels inside region: {path}")

        units = rainfall.attrs.get("units", "unknown")

        return {
            "date": extract_date(path, dataset),
            "source_file": path.name,
            "variable": variable,
            "units": units,
            "pixel_count": int(valid.size),
            "mean": round(float(np.mean(valid)), 4),
            "maximum": round(float(np.max(valid)), 4),
            "minimum": round(float(np.min(valid)), 4),
            "median": round(float(np.median(valid)), 4),
            "sum_of_pixel_values": round(float(np.sum(valid)), 4),
        }


def process_region(name, config):
    folder = config["input"]
    boundary_path = config["boundary"]

    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder missing: {folder}")
    if not boundary_path.is_file():
        raise FileNotFoundError(f"Boundary file missing: {boundary_path}")

    files = sorted(folder.glob("*.nc4"))
    if not files:
        raise FileNotFoundError(f"No .nc4 files found in {folder}")

    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")

    daily = []
    failures = []

    for path in files:
        try:
            result = process_file(path, boundary)
            daily.append(result)
            print(f"  OK {result['date']}: {result['mean']} {result['units']}")
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})
            print(f"  FAILED {path.name}: {exc}")

    daily.sort(key=lambda item: item["date"])

    if not daily:
        raise RuntimeError(f"No valid daily files processed for {name}")

    output = {
        "id": f"{name}-gpm-daily-rainfall",
        "region": name,
        "dataset": "NASA GPM IMERG Daily Late Run",
        "collection": "GPM_3IMERGDL",
        "version": "07",
        "processing_level": "Late Run",
        "temporal_resolution": "daily",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(folder.relative_to(ROOT)),
        "boundary": str(boundary_path.relative_to(ROOT)),
        "daily_observations": daily,
        "successful_files": len(daily),
        "failed_files": failures,
        "quality_note": (
            "Rainfall observations are contextual precipitation estimates, "
            "not direct flood extent measurements."
        ),
    }

    config["output"].parent.mkdir(parents=True, exist_ok=True)
    config["output"].write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"  Saved: {config['output'].relative_to(ROOT)}")
    print(f"  Successful: {len(daily)} | Failed: {len(failures)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        choices=["feni", "south_chattogram", "all"],
        default="all",
    )
    args = parser.parse_args()

    regions = (
        REGIONS.items()
        if args.region == "all"
        else [(args.region, REGIONS[args.region])]
    )

    for name, config in regions:
        print(f"\nProcessing {name}...")
        process_region(name, config)

    print("\nGPM processing finished.")


if __name__ == "__main__":
    main()
