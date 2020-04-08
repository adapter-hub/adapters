import logging
import json
from hashlib import sha1
from os import path
from transformers.file_utils import is_remote_url
import requests


logger = logging.getLogger(__name__)

CONFIG_INDEX_FILE = "index.json"

def urljoin(*args):
    return '/'.join([s.strip('/') for s in args])

def get_config_hash(*configs):
    h = sha1()
    for config in configs:
        dict_str = json.dumps(config, sort_keys=True)
        h.update(dict_str.encode(encoding='utf-8'))
    return h.hexdigest()

def http_get_config_index(url):
    response = requests.get(urljoin(url, CONFIG_INDEX_FILE))
    if response.status_code == 200:
        return response.json()
    else:
        return None

def find_matching_config_path(url, config):
    index_dict = http_get_config_index(url)
    if index_dict:
        # TODO use a default config
        assert config['config'], "Multiple adapter configs available. Specify a configuration in advance."
        config_hash = get_config_hash(config)
        if config_hash in index_dict:
            # we found a matching configuration in the list
            resolved_path = urljoin(url, index_dict[config_hash])
            logger.info("found matching pretrained adapter at {}".format(resolved_path))
            return resolved_path
        else:
            raise EnvironmentError("No adapter matching the current configuration found at {}".format(url))
    else:
        # if there is no index file, just return the base directory
        logger.info("no configuration index found at {}, using default".format(url))
        return url
