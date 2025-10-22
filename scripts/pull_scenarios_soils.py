
import argparse
from pathlib import Path
import time
import yaml
from generate_soilgrids_soil_file import generate_soil_file
from cropgymzoo import _SCENARIO_PATH, _SOILGRIDS_PATH


def generate_soil_files(longitude: float, latitude: float):
    generate_soil_file(longitude, latitude)
    print(f" -> generating soil file for ({latitude:.5f}, {longitude:.5f})")

def parse_args():
    p = argparse.ArgumentParser(description="Generate soil files for 2020 YAMLs (skip if existing).")
    p.add_argument("--root", default="scenarios")
    p.add_argument("--year", default="2020")
    p.add_argument("--soildir", default="soils")
    p.add_argument("--glob", default="farmer_*.yaml")
    p.add_argument("--sleep", type=float, default=0.0)
    return p.parse_args()

def iter_yaml_paths(root: Path, year: str, pattern: str):
    for region_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ydir = region_dir / year
        if not ydir.is_dir():
            continue
        for f in sorted(ydir.glob(pattern)):
            yield f

def load_yaml(fp: Path) -> dict:
    with fp.open("r") as f:
        return yaml.safe_load(f) or {}

def iter_field_coords(doc: dict):
    for k, v in doc.items():
        if isinstance(v, dict) and k.startswith("field-"):
            lon, lat = v.get("soil_lon"), v.get("soil_lat")
            if lon is not None and lat is not None:
                yield float(lon), float(lat), str(k)

def trim4(x: float) -> str:
    """Format to 4 decimals then drop trailing zeros and dot."""
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"

def filename_variants(lon: float, lat: float) -> list[str]:
    lon4, lat4 = f"{lon:.4f}", f"{lat:.4f}"
    lonstr, latstr = trim4(lon), trim4(lat)
    return [
        f"soil_{lon4}_{lat4}.yaml",
        f"soil_{lonstr}_{lat4}.yaml",
        f"soil_{lon4}_{latstr}.yaml",
        f"soil_{lonstr}_{latstr}.yaml",
    ]

def main():
    args = parse_args()
    root = Path(_SCENARIO_PATH)
    soildir = Path(_SOILGRIDS_PATH)

    made = skipped = failed = 0
    seen_keys: set[str] = set()
    failed_keys = []

    for yaml_path in iter_yaml_paths(root, args.year, args.glob):
        data = load_yaml(yaml_path)
        for lon, lat, k in iter_field_coords(data):
            key = f"{lon:.4f},{lat:.4f}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            variants = [soildir / name for name in filename_variants(lon, lat)]
            if any(p.exists() for p in variants):
                print(f"----- exists → skip  ({lon:.4f}, {lat:.4f})")
                skipped += 1
                continue

            try:
                generate_soil_file(lon, lat)
                made += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as e:
                print(f"!!!!  failed for ({lon:.4f}, {lat:.4f}): {e}")
                failed += 1
                failed_keys.append((key,k,yaml_path))

    print("\n--- Summary ---")
    print(f"Generated : {made}")
    print(f"Skipped   : {skipped} (existing)")
    print(f"Failed    : {failed}")
    if failed_keys:
        print("\nFailed keys:")
        for key in failed_keys:
            print(f"  {key}")

if __name__ == "__main__":
    main()