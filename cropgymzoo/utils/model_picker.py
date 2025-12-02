import torch

from cropgymzoo.run_scenarios import region_crop_picker
from cropgymzoo.utils.scenario_utils import get_scenario_based_on_loc


def model_picker(model_file, dict_fields):
    orig_model_dict = torch.load(model_file, weights_only=False)

    assert isinstance(orig_model_dict, dict)

    new_model_dict = {}
    new_obs_rms_dict = {}
    for name, field in dict_fields.items():
        crop = field['crop']
        coor = (field['soil_lat'], field['soil_lon'])
        region = get_scenario_based_on_loc(coor)

        orig_agent_name = region_crop_picker(region, crop)
        new_model_dict[name] = orig_model_dict["models"][orig_agent_name]
        if isinstance(orig_model_dict["obs_rms"], dict):
            new_obs_rms_dict[name] = orig_model_dict["obs_rms"][orig_agent_name]

    orig_model_dict['models'] = new_model_dict
    if new_obs_rms_dict:
        orig_model_dict['obs_rms'] = new_obs_rms_dict

    return orig_model_dict
