from pathlib import Path

import yaml

# project directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '../data'
OUTPUT_DIR = BASE_DIR / 'output'


def _parse_config():
    class Struct(object):
        def __init__(self, di):
            for a, b in di.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, Struct(b) if isinstance(b, dict) else b)

    with open(BASE_DIR / 'configs/config.yaml', 'r') as stream:
        try:
            return Struct(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


configs = _parse_config()
