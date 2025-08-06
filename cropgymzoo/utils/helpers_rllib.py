import os
import yaml

from cropgymzoo import _FIELDS_CONFIG

def get_agent_ids() -> list[str]:
    with open(_FIELDS_CONFIG) as f:
        dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

    return list(dict_fields.keys())