import os
import yaml
import logging.config
import logging
import coloredlogs
from backend.utils.constants import PROJECT_ROOT

def setup_logging(default_path=PROJECT_ROOT/'logging.yaml', default_level=logging.NOTSET):
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')
