"""
Build ap_mandals.geojson from Census 2011 subdistricts (mandal-equivalent),
clipped to current Andhra Pradesh using ap_boundary.geojson.

1) Downloads SubDistricts_2011.parquet if missing (ramSeraph/indian_admin_boundaries, CC0).
2) Keeps features whose centroid lies inside the AP state outline.
3) Simplifies geometry and writes GeoJSON for the Leaflet map.

Run: python build_ap_mandals.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).parent
PARQUET = ROOT / "SubDistricts_2011.parquet"
PARQUET_URL = (
    "https://github.com/ramSeraph/indian_admin_boundaries/releases/download/"
    "census-2011/SubDistricts_2011.parquet"
)
BOUNDARY = ROOT / "ap_boundary.geojson"
OUT = ROOT / "ap_mandals.geojson"
SIMPLIFY_DEG = 0.00012  # ~12 m; increase for smaller files


def ensure_parquet() -> None:
    if PARQUET.exists():
        return
    print(f"Downloading {PARQUET_URL} …")
    urllib.request.urlretrieve(PARQUET_URL, PARQUET)
    print(f"Saved {PARQUET}")


def main() -> None:
    ensure_parquet()
    if not BOUNDARY.exists():
        raise SystemExit(f"Missing {BOUNDARY.name}; run build_ap_boundary.py first.")

    sub = gpd.read_parquet(PARQUET)
    ap = gpd.read_file(BOUNDARY)
    if sub.crs != ap.crs:
        ap = ap.to_crs(sub.crs)

    # Project for reliable centroid-within test (geographic CRS centroids are biased)
    proj_crs = "EPSG:3857"
    boundary = ap.to_crs(proj_crs).geometry.union_all()
    centroids = sub.to_crs(proj_crs).geometry.centroid
    ap_m = sub[centroids.within(boundary)].copy()
    if ap_m.empty:
        raise SystemExit("No subdistricts inside boundary (CRS mismatch?).")

    ap_m["geometry"] = ap_m.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)

    keep = ap_m[
        ["stname", "dtname", "sdtname", "stcode11", "dtcode11", "sdtcode11", "geometry"]
    ].rename(
        columns={
            "stname": "state",
            "dtname": "district",
            "sdtname": "mandal",
            "stcode11": "st_code",
            "dtcode11": "dt_code",
            "sdtcode11": "mandal_code",
        }
    )
    keep["source"] = "Census 2011 subdistricts (ramSeraph/indian_admin_boundaries, CC0)"

    keep.to_file(OUT, driver="GeoJSON")
    # Minify one line for slightly smaller transfer (optional)
    data = json.loads(OUT.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    print(f"Wrote {OUT} — {len(keep)} mandals / subdistricts, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
