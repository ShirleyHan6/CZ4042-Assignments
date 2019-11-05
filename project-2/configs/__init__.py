import os.path as osp
import sys
from typing import Union

import yaml

# project directory
BASE_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    '..'
)

# add PYTHONPATH
sys.path.extend([BASE_DIR])


def _parse_config() -> Union[dict, list, None]:
    with open(osp.join(BASE_DIR, 'config', 'config.yaml'), 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


configs = _parse_config()


__all__ = ["BASE_DIR", "configs"]
