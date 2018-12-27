import logging.config
import yaml
from pkg_resources import resource_stream


def setup_logging():
    # with resource_stream('resources', 'logging.yaml') as f:
    #     logging.config.dictConfig(yaml.safe_load(f.read()))
    with open('../config/logging.yaml','r') as f:
        logging.config.dictConfig(yaml.load(f))
