import logging
import json
from hashlib import sha1
import requests


logger = logging.getLogger(__name__)


CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
HEAD_WEIGHTS_NAME = "pytorch_adapter_head.bin"


def urljoin(*args):
    return '/'.join([s.strip('/') for s in args])


def get_config_hash(*configs):
    h = sha1()
    for config in configs:
        dict_str = json.dumps(config, sort_keys=True)
        h.update(dict_str.encode(encoding='utf-8'))
    return h.hexdigest()


def remote_file_exists(url):
    r = requests.head(url)
    return r.status_code == 200


def find_matching_config_path(url, config, version=None):
    # if there is a module saved in the root dir just use this
    if remote_file_exists(urljoin(url, CONFIG_NAME)):
        logger.info("found a default pretrained adapter at {}".format(url))
        return url
    # otherwise search for an exact match based on the config
    else:
        assert config['config'], "Multiple adapter configs available. Specify a configuration in advance."
        config_hash = get_config_hash(config)
        if remote_file_exists(urljoin(url, config_hash, "")):
            # if we specified a version, search if it's available
            if version:
                if remote_file_exists(urljoin(url, config_hash, str(version), "")):
                    resolved_path = urljoin(url, config_hash, str(version), "")
                else:
                    resolved_path = urljoin(url, config_hash, "")
                    logger.warn("version {} of adapter not found, falling back to default".format(version))
            else:
                resolved_path = urljoin(url, config_hash, "")
            logger.info("found matching pretrained adapter at {}".format(resolved_path))
            return resolved_path
        else:
            raise EnvironmentError("No adapter matching the current configuration found at {}".format(url))
