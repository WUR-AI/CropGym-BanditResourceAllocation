import re
from pathlib import Path
import sys
import yaml
from cropgymzoo import _SCENARIO_PATH

# --- config ---------------------------------------------------------------
ROOT = Path(_SCENARIO_PATH)
REGIONS = ["gelderland", "zeeland", "groningen"]  # restrict; set to [] to include all
YAML_GLOB = "farmer_*.yaml"
FIELD_KEY_RE = re.compile(r"^field-\d+$")
OUT_YAML = "scenario_coords.yaml"
ROUND_DP = 4  # round for de-duplication and neat output
# -------------------------------------------------------------------------

def load_yaml(fp: Path) -> dict:
    try:
        with fp.open("r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"[WARN] Could not read {fp}: {e}", file=sys.stderr)
        return {}

def iter_field_coords(doc: dict):
    """Yield (lon, lat) for keys like field-1, field-2, ..."""
    if not isinstance(doc, dict):
        return
    for k, v in doc.items():
        if FIELD_KEY_RE.match(str(k)) and isinstance(v, dict):
            lon = v.get("soil_lon")
            lat = v.get("soil_lat")
            if lon is not None and lat is not None:
                yield float(lon), float(lat)

def main():
    # region -> set of (lon, lat) tuples (rounded)
    coords_by_region: dict[str, set[tuple[float, float]]] = {}

    for region_dir in ROOT.iterdir():
        if not region_dir.is_dir():
            continue
        region = region_dir.name
        if REGIONS and region not in REGIONS:
            continue

        # scan all years under this region
        for year_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            for f in sorted(year_dir.glob(YAML_GLOB)):
                data = load_yaml(f)
                for lon, lat in iter_field_coords(data):
                    lon_r = round(lon, ROUND_DP)
                    lat_r = round(lat, ROUND_DP)
                    coords_by_region.setdefault(region, set()).add((lon_r, lat_r))

    # turn sets into sorted lists for stable YAML output
    coords_out: dict[str, list[list[float]]] = {}
    for region, s in coords_by_region.items():
        # sort by lon then lat
        pairs = sorted(list(s), key=lambda t: (t[0], t[1]))
        coords_out[region] = [[lon, lat] for lon, lat in pairs]

    if not coords_out:
        print("No coordinates found. Check ROOT/REGIONS/globs.")
        return

    # write YAML
    with open(OUT_YAML, "w") as fh:
        yaml.safe_dump(coords_out, fh, sort_keys=True, default_flow_style=False)

    # brief summary
    print(f"Wrote {OUT_YAML}")
    for region, pairs in coords_out.items():
        print(f" - {region}: {len(pairs)} unique coords (rounded to {ROUND_DP} dp)")

if __name__ == "__main__":
    main()