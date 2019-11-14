from pathlib import Path

import yaml

# project directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '../data'
OUTPUT_DIR = BASE_DIR / 'output'


class DictToObject(object):
    def __init__(self, di):
        for a, b in di.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictToObject(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, DictToObject(b) if isinstance(b, dict) else b)


def parse_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return DictToObject(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
