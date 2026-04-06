"""
Merge all features in geojson_ap.geojson into one Andhra Pradesh outline,
simplify for a smaller file, write ap_boundary.geojson.
Run once from the project folder: python build_ap_boundary.py
"""
import json
from pathlib import Path

from shapely.geometry import shape, mapping
from shapely.ops import unary_union

SRC = Path(__file__).parent / "geojson_ap.geojson"
OUT = Path(__file__).parent / "ap_boundary.geojson"
# ~90 m at this latitude; keeps outline smooth but much lighter than raw union
SIMPLIFY_TOLERANCE = 0.0008


def main() -> None:
    with SRC.open(encoding="utf-8") as f:
        fc = json.load(f)

    geoms = []
    for feat in fc.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        geoms.append(shape(g))

    if not geoms:
        raise SystemExit("No geometries found in geojson_ap.geojson")

    merged = unary_union(geoms)
    merged = merged.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)

    out_fc = {
        "type": "FeatureCollection",
        "name": "Andhra Pradesh state boundary (dissolved from districts)",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Andhra Pradesh",
                    "st_nm": "Andhra Pradesh",
                    "source": "dissolved from geojson_ap.geojson",
                },
                "geometry": mapping(merged),
            }
        ],
    }

    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out_fc, f, separators=(",", ":"))

    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
