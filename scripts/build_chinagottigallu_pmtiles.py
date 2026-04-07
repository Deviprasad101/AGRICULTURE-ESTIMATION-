"""
Build data/chinagottigallu_fields.pmtiles (MVT) from data/chinagottigallu_fields.geojson.

Requires: pip install mapbox-vector-tile mercantile shapely pmtiles
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from pathlib import Path

import mapbox_vector_tile
import mercantile
from mapbox_vector_tile.encoder import on_invalid_geometry_make_valid
from pmtiles.convert import mbtiles_to_pmtiles
from shapely.geometry import box, mapping, shape
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = ROOT / "data" / "chinagottigallu_fields.geojson"
MBTILES_PATH = ROOT / "data" / "chinagottigallu_fields.mbtiles"
PMTILES_PATH = ROOT / "data" / "chinagottigallu_fields.pmtiles"

ZOOMS = (12, 13, 14, 15, 16)
LAYER_NAME = "fields"


def _safe_intersection(feat: dict, tile: mercantile.Tile) -> dict | None:
    geom = shape(feat["geometry"])
    if not geom.is_valid:
        geom = make_valid(geom)
    b = mercantile.bounds(tile)
    rect = box(b.west, b.south, b.east, b.north)
    if not geom.intersects(rect):
        return None
    inter = geom.intersection(rect)
    if inter.is_empty:
        return None
    try:
        g = mapping(inter)
    except Exception:
        return None
    if g["type"] not in ("Polygon", "MultiPolygon"):
        return None
    return {"type": "Feature", "geometry": g, "properties": feat.get("properties") or {}}


def main() -> int:
    if not GEOJSON_PATH.is_file():
        print("Missing", GEOJSON_PATH, file=sys.stderr)
        return 1

    with open(GEOJSON_PATH, encoding="utf-8") as f:
        fc = json.load(f)
    feats = fc.get("features") or []
    if not feats:
        print("No features in GeoJSON", file=sys.stderr)
        return 1

    # World bounds for metadata
    min_lon, min_lat, max_lon, max_lat = 180.0, 90.0, -180.0, -90.0
    for feat in feats:
        g = shape(feat["geometry"])
        x0, y0, x1, y1 = g.bounds
        min_lon = min(min_lon, x0)
        min_lat = min(min_lat, y0)
        max_lon = max(max_lon, x1)
        max_lat = max(max_lat, y1)

    MBTILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MBTILES_PATH.exists():
        MBTILES_PATH.unlink()
    if PMTILES_PATH.exists():
        PMTILES_PATH.unlink()

    conn = sqlite3.connect(str(MBTILES_PATH))
    cur = conn.cursor()
    cur.execute("CREATE TABLE metadata (name text, value text);")
    cur.execute(
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);"
    )

    meta_json = {
        "vector_layers": [
            {
                "id": LAYER_NAME,
                "fields": {
                    "field_id": "String",
                    "crop": "String",
                    "area_ac": "Number",
                    "farmer": "String",
                    "season": "String",
                    "yield": "String",
                    "survey": "String",
                    "village": "String",
                    "mandal": "String",
                    "district": "String",
                },
            }
        ]
    }
    cur.executemany(
        "INSERT INTO metadata VALUES (?,?);",
        [
            ("name", "Chinagottigallu fields"),
            ("format", "pbf"),
            ("minzoom", str(min(ZOOMS))),
            ("maxzoom", str(max(ZOOMS))),
            ("bounds", f"{min_lon},{min_lat},{max_lon},{max_lat}"),
            (
                "center",
                f"{(min_lon + max_lon) / 2:.7f},{(min_lat + max_lat) / 2:.7f},{max(ZOOMS)}",
            ),
            ("json", json.dumps(meta_json)),
        ],
    )

    tile_count = 0
    for z in ZOOMS:
        for tile in mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=[z]):
            tile_feats = []
            for feat in feats:
                clipped = _safe_intersection(feat, tile)
                if clipped:
                    tile_feats.append(
                        {
                            "geometry": clipped["geometry"],
                            "properties": clipped["properties"],
                        }
                    )
            if not tile_feats:
                continue
            b = mercantile.bounds(tile)
            q = (b.west, b.south, b.east, b.north)
            try:
                pbf = mapbox_vector_tile.encode(
                    [{"name": LAYER_NAME, "features": tile_feats}],
                    default_options={
                        "quantize_bounds": q,
                        "extents": 4096,
                        "on_invalid_geometry": on_invalid_geometry_make_valid,
                    },
                )
            except Exception as e:
                print(f"Skip tile {tile}: {e}", file=sys.stderr)
                continue
            data = gzip.compress(pbf)
            tms_y = (1 << z) - 1 - tile.y
            cur.execute(
                "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?,?,?,?);",
                (z, tile.x, tms_y, data),
            )
            tile_count += 1

    conn.commit()
    conn.close()
    print(f"Wrote {tile_count} tiles to {MBTILES_PATH.name}")

    mbtiles_to_pmtiles(str(MBTILES_PATH), str(PMTILES_PATH), None)
    MBTILES_PATH.unlink(missing_ok=True)
    print(f"Wrote {PMTILES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
