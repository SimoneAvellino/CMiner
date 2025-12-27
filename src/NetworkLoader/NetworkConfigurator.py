import os
import json
from   os.path import join
from   pathlib import Path


class NetworkConfigurator:
    def __init__(self, network_path, network_type):
        self.network_path = network_path
        self.network_type = network_type
        self.config       = self.configuration_reading()

    def configuration_reading(self):
        root_dir    = Path(os.path.abspath(__file__)).parents[1]
        config_file = join(root_dir, "../Configurations", "LoadingFiles", self.network_type.lower() + "_conf.json")
        with open(config_file, 'r') as file:
            config = json.load(file)
            # split path in file_path and file_name
            config["file_path"], config["file_name"] = os.path.split(self.network_path)
            return config