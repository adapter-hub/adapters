import logging
import json
from filelock import FileLock
import hashlib
import os
from os.path import join
from pathlib import Path
import requests
import shutil
import tarfile
from zipfile import ZipFile, is_zipfile
from .file_utils import is_remote_url, get_from_cache, torch_cache_home


logger = logging.getLogger(__name__)

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
HEAD_WEIGHTS_NAME = "pytorch_adapter_head.bin"

ADAPTER_IDENTIFIER_PATTERN = r"[a-zA-Z\-_]{2,}"
ADAPTER_HUB_URL = "http://adapter-hub.webredirect.org/repo/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "index.json"

# these keys of the adapter config are used to calculate the config hash
ADAPTER_CONFIG_HASH_KEYS = [
    'config', 'hidden_size', 'model', 'type'
]

# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")


def urljoin(*args):
    return '/'.join([s.strip('/') for s in args])


def get_adapter_config_hash(config):
    """Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The SHA-1 hash of the config dict.
    """
    h = hashlib.sha1()
    # only hash non-empty/ true items and required keys
    config = {
        k: v for (k, v) in config.items() if v and k in ADAPTER_CONFIG_HASH_KEYS
    }
    dict_str = json.dumps(config, sort_keys=True)
    h.update(dict_str.encode(encoding='utf-8'))
    return h.hexdigest()


def remote_file_exists(url):
    r = requests.head(url)
    return r.status_code == 200


def download_cached(url, checksum=None, checksum_algo='sha1', cache_dir=None, force_extract=False, **kwargs):
    if not cache_dir:
        cache_dir = ADAPTER_CACHE
    if isinstance(url, Path):
        url = str(url)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url):
        output_path = get_from_cache(url, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError("Unable to parse '{}' as a URL".format(url))

    # if checksum is given, verify it
    if checksum and checksum_algo:
        h = hashlib.new(checksum_algo)
        with open(output_path, 'rb') as f:
            h.update(f.read())
        calculated_checksum = h.hexdigest()
        if calculated_checksum != checksum.lower():
            raise EnvironmentError("Failed to verify checksum of '{}'".format(output_path))

    if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
        return output_path

    # Path where we extract compressed archives
    # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
        return output_path_extracted

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_path_extracted)
                zip_file.close()
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

    return output_path_extracted


def find_in_index(identifier: str, config: dict):
    index_file = download_cached(ADAPTER_HUB_INDEX_FILE)
    if not index_file:
        raise EnvironmentError("Unable to load adapter hub index file. The file might be temporarily unavailable.")
    with open(index_file, 'r') as f:
        adapter_index = json.load(f)
    if identifier in adapter_index:
        index_entry = adapter_index[identifier]
        # now search for an entry matching the given config
        if config:
            assert config['config'], "Specify an adapter configuration to search for."
            config_hash = get_adapter_config_hash(config)
            if config_hash in index_entry['adapters']:
                hub_entry = index_entry['adapters'][config_hash]
                logger.info("Found matching adapter at: {}".format(hub_entry))
                return hub_entry
        # we haven't found a perfect match, either because no config was given or the config was not found
        if 'default' in index_entry:
            logger.warn("No adapter matching the given config found. Falling back to default.")
            return index_entry['default']
        else:
            raise EnvironmentError(
                "Not matching config found for adapter '{}'. Please give a valid config.".format(identifier)
            )
    else:
        raise EnvironmentError("No adapter with name '{}' was found in the adapter index.".format(identifier))


def http_get_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise EnvironmentError("Failed to get file {}".format(url))


def pull_from_hub(adapter_name: str, config: dict, version: str = None, **kwargs):
    # first, search the correct entry in the index
    hub_entry_url = find_in_index(adapter_name, config)
    hub_entry = http_get_json(hub_entry_url)['_meta']

    # set version
    if not version:
        version = hub_entry['default_version']
    elif version not in hub_entry['files']:
        logger.warn(
            "Version '{}' of adapter '{}' not found. Falling back to default.".format(version, adapter_name)
        )
        version = hub_entry['default_version']
    file_entry = hub_entry['files'][version]

    # start downloading
    logger.info("Resolved adapter files at {}.".format(file_entry['url']))
    download_path = download_cached(file_entry['url'], checksum=file_entry['sha1'], **kwargs)
    if not download_path:
        raise EnvironmentError(
            "Unable to load file from {}. The file might be unavailable.".format(file_entry['url'])
        )
    return download_path
