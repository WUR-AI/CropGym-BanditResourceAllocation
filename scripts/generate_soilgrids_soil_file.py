import argparse
import os
import sys
from cropgymzoo import _BASE_PATH

from cropgymzoo.utils_soil.soilgrids import get_df_soilgrids
from cropgymzoo.utils_soil.generate_soil_files import (
    calculate_van_genuchten,
    generate_df_soil_input,
    generate_soil_yaml,
    dump_soil_yaml
)

soil_save_dir = os.path.join(_BASE_PATH, "cropgymzoo", "configs", "soil", "soilgrids")

def generate_soil_file(longitude, latitude):
    """
    Generate a YAML soil file for given longitude and latitude.

    Args:
        longitude (float): Longitude value.
        latitude (float): Latitude value.
    """

    # Get soil data from SoilGrids™ based on longitude and latitude
    soil_data = get_df_soilgrids(lon=longitude, lat=latitude)

    vg_data = calculate_van_genuchten(soil_data)

    df_soil_input = generate_df_soil_input(vg_data)

    soil_yaml = generate_soil_yaml(df_soil_input)

    # Write the soil data to a YAML file
    path_file = os.path.join(soil_save_dir, f"soil_{longitude}_{latitude}.yaml")
    dump_soil_yaml(soil_yaml, path_file)

    print(f"YAML soil file has been created at {path_file}.")


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print("Usage: python generate_soil_file.py --lon <longitude> --lat <latitude>")
        print("Example: python generate_soil_file.py 6.656 52.966")
        sys.exit(1)


    parser = argparse.ArgumentParser(description="Generate a YAML soil file for a given longitude and latitude.")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, help="Longitude for the soil data.")
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, help="Latitude for the soil data.")

    args = parser.parse_args()

    # Validate the output directory
    output_dir = os.path.dirname(soil_save_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the YAML file
    generate_soil_file(args.longitude,
                       args.latitude)


if __name__ == "__main__":
    main()
