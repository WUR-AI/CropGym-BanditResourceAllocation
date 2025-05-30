import os

_SOURCE_PATH = os.path.dirname(os.path.realpath(__file__))
_BASE_PATH = os.path.dirname(_SOURCE_PATH)
_CONFIG_PATH = os.path.join(_SOURCE_PATH, 'configs')

_CROPS_PATH = os.path.join(_CONFIG_PATH, 'crop')
_SITE_PATH = os.path.join(_CONFIG_PATH, 'sites')
_SOIL_PATH = os.path.join(_CONFIG_PATH, 'soil')

_CROPS_LIST = os.path.join(_CROPS_PATH, 'crops.yaml')
_AGRO_CALENDAR_CONFIG = os.path.join(_CONFIG_PATH, 'agro', 'generic_cropcalendar.yaml')
_FIELDS_CONFIG = os.path.join(_CONFIG_PATH, 'fields.yaml')
_CROPS_CONFIG = os.path.join(_CONFIG_PATH, 'crop_info.yaml')
_WOFOST_CONFIG = os.path.join(_CONFIG_PATH, 'Wofost81_NWLP_MLWB_SNOMIN.conf')

from cropgymzoo.envs.pcse_env import PCSEEnv
from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.worker_env import ParallelRLWorkers
from cropgymzoo.envs.allocation_env import AllocationBandit