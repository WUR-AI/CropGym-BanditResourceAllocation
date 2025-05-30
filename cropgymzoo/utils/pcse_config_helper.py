import os
import pcse
from cropgymzoo import _SOURCE_PATH

def get_pcse_args(
        crop: int = None,
        soil: int =None,
        site=None,
)->dict:
    kwargs = {
        'model_config': os.path.join(_SOURCE_PATH, 'Wofost81_NWLP_MLWB_SNOMIN.conf'),
        'agro_config': os.path.join(_SOURCE_PATH, 'agro'),
        'crop_parameters': os.path.join(_SOURCE_PATH, 'crop'),
        'soil_parameters': os.path.join(_SOURCE_PATH, 'soil'),
        'site_parameters': os.path.join(_SOURCE_PATH, 'site'),
    }
    return kwargs

def get_crop_id(
        index: int = 0,
):
    d = {
        0: 'winterwheat',
        1: 'potato',
        2: 'sugarbeet'
    }
    return d[index]

def get_soil_coordinates_by_id_lt(
        index: int = 0,
):
    d = {
        0: (23.91, 55.12),
        1: (23.89, 55.13),
        2: (23.9, 55.11),
        3: (23.89, 55.11),
        4: (23.88, 55.11),
        5: (23.87, 55.12),
    }
    return d[index]

def get_soil_coordinates_by_id_nl(
        index: int = 0,
):
    d = {
        0: (5.54, 52.49),
        1: (5.54, 52.5),
        2: (5.55, 52.5),
        3: (5.56, 52.5),
        4: (5.57, 52.5),
        5: (5.58, 52.5),
        6: (5.58, 52.6),
    }
    return d[index]

def get_soil_files_by_coordinates(
        coordinates: tuple,
):
    return f"soil_{coordinates[0]}_{coordinates[1]}.yaml"

