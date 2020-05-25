import logging
import json
from filelock import FileLock
import hashlib
import os
from os.path import join, isdir, isfile
from pathlib import Path
import re
import requests
import shutil
import tarfile
from typing import Optional, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile
from .file_utils import is_remote_url, get_from_cache, torch_cache_home
from .adapters_config import ADAPTER_CONFIG_MAP, AdapterType, build_full_config
from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
HEAD_WEIGHTS_NAME = "pytorch_adapter_head.bin"

ADAPTER_IDENTIFIER_PATTERN = r"[a-zA-Z\-_\/@]{2,}"
ADAPTER_HUB_URL = "https://raw.githubusercontent.com/calpt/nothing-to-see-here/master/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "dist/adapters_{}.json"
ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + "dist/architectures.json"

# these keys of the adapter config are used to calculate the config hash
ADAPTER_CONFIG_HASH_SECTIONS = [
    (['hidden_size', 'model_type', 'type'], 8), (['config'], 16)
]

# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")


def urljoin(*args):
    return '/'.join([s.strip('/') for s in args])


def _minimize_dict(d):
    if isinstance(d, dict):
        return {k: _minimize_dict(v) for (k, v) in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config):
    """Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    config_hash = ""
    for keys, length in ADAPTER_CONFIG_HASH_SECTIONS:
        config_section = {k: _minimize_dict(v) for (k, v) in config.items() if k in keys}
        if config_section:
            dict_str = json.dumps(config_section, sort_keys=True)
            h = hashlib.sha1()
            h.update(dict_str.encode(encoding='utf-8'))
            config_hash += h.hexdigest()[:length]
        else:
            config_hash += "x" * length
    return config_hash


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

    if not output_path:
        return None

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


def resolve_adapter_config(config: Union[dict, str]):
    # already a dict, so we don't have to do anything
    if isinstance(config, dict):
        return config
    # first, look in local map
    if config in ADAPTER_CONFIG_MAP:
        return ADAPTER_CONFIG_MAP[config]
    # now, try to find in hub index
    index_file = download_cached(ADAPTER_HUB_CONFIG_FILE)
    if not index_file:
        raise EnvironmentError("Unable to load adapter hub index file. The file might be temporarily unavailable.")
    with open(index_file, 'r') as f:
        config_index = json.load(f)
    if config in config_index:
        return config_index[config]
    else:
        raise ValueError("Could not identify '{}' as a valid adapter configuration.".format(config))


def _split_identifier(identifier):
    task, subtask, org_name = None, None, None
    identifier = identifier.split("@")
    if len(identifier) > 1:
        org_name = identifier[1]
    identifier = identifier[0].split("/")
    if len(identifier) > 1:
        subtask = identifier[1]
    task = identifier[0]
    return task, subtask, org_name


def _dict_extract(d, primary_key, secondary_key=None):
    for k, v in d.items():
        if k == primary_key:
            if secondary_key:
                if secondary_key in v.keys():
                    yield v[secondary_key]
            else:
                for k, v in v.items():
                    yield v
        else:
            for k, v in v.items():
                if k == primary_key:
                    yield v


def find_in_index(
        identifier: str,
        adapter_config: dict,
        adapter_type: AdapterType,
        model_config: PretrainedConfig) -> Optional[str]:
    config = build_full_config(adapter_config, adapter_type, model_config)
    index_file = download_cached(ADAPTER_HUB_INDEX_FILE.format(config['type']))
    if not index_file:
        raise EnvironmentError("Unable to load adapter hub index file. The file might be temporarily unavailable.")
    with open(index_file, 'r') as f:
        adapter_index = json.load(f)
    # split into <task>/<subtask>@<org>
    task, subtask, org = _split_identifier(identifier)
    # find all entries for this task and subtask
    entries = list(_dict_extract(adapter_index, task, subtask))
    if not entries:
        # we found no matching entry
        return None
    elif len(entries) == 1:
        index_entry = entries[0]
    else:
        # there are multiple possible options for this identifier
        raise ValueError("Found multiple possible adapters matching '{}'.".format(identifier))
    # go on with searching a matching config hash in the task entry
    assert config['config'], "Specify an adapter configuration to search for."
    config_hash = get_adapter_config_hash(config)
    if config_hash in index_entry:
        # now match the org if given
        version = org or index_entry[config_hash]["default"]
        hub_entry = index_entry[config_hash]['versions'].get(version, None)
        if hub_entry:
            logger.info("Found matching adapter at: {}".format(hub_entry))
        return hub_entry
    else:
        raise ValueError(
            "No adapter '{}' found for the current model or configuration.".format(identifier)
        )


def http_get_json(url):
    # check if it's a relative url
    if not urlparse(url).netloc:
        url = urljoin(ADAPTER_HUB_URL, url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise EnvironmentError("Failed to get file {}".format(url))


def pull_from_hub(
        specifier: str,
        adapter_config: Union[dict, str],
        adapter_type: AdapterType,
        model_config: PretrainedConfig,
        version: str = None,
        **kwargs) -> str:
    # resolve config if it's an identifier
    adapter_config = resolve_adapter_config(adapter_config)
    # search the correct entry in the index
    hub_entry_url = find_in_index(specifier, adapter_config, adapter_type, model_config)
    if not hub_entry_url:
        raise EnvironmentError("No adapter with name '{}' was found in the adapter index.".format(specifier))
    hub_entry = http_get_json(hub_entry_url)['_meta']

    # set version
    if not version:
        version = hub_entry['default_version']
    elif version not in hub_entry['files']:
        logger.warn(
            "Version '{}' of adapter '{}' not found. Falling back to default.".format(version, specifier)
        )
        version = hub_entry['default_version']
    file_entry = hub_entry['files'][version]

    # start downloading
    logger.info("Resolved adapter files at {}.".format(file_entry['url']))
    # TODO add support for other checksums
    download_path = download_cached(file_entry['url'], checksum=file_entry['sha1'], **kwargs)
    if not download_path:
        raise EnvironmentError(
            "Unable to load file from {}. The file might be unavailable.".format(file_entry['url'])
        )
    return download_path


def resolve_adapter_path(
        adapter_name_or_path,
        adapter_config: Union[dict, str],
        adapter_type: AdapterType,
        model_config: PretrainedConfig,
        version: str = None,
        **kwargs) -> str:
    # url of a folder containing pretrained adapters -> try to load from this url
    if is_remote_url(adapter_name_or_path):
        resolved_folder = download_cached(adapter_name_or_path, **kwargs)
        if not resolved_folder:
            raise EnvironmentError(
                "Unable to load file from {}. The file might be unavailable.".format(resolved_folder)
            )
        return resolved_folder
    # path to a local folder saved using save()
    elif isdir(adapter_name_or_path):
        if isfile(join(adapter_name_or_path, WEIGHTS_NAME)) and isfile(join(adapter_name_or_path, CONFIG_NAME)):
            return adapter_name_or_path
        else:
            raise EnvironmentError(
                "No file {} or no file {} found in directory {}".format(
                    WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path)
            )
    # matches possible form of identifier in hub
    elif re.fullmatch(ADAPTER_IDENTIFIER_PATTERN, adapter_name_or_path):
        return pull_from_hub(
            adapter_name_or_path, adapter_config, adapter_type, model_config, version=version, **kwargs
        )
    else:
        raise ValueError("Unable to identify {} as a valid module location.".format(adapter_name_or_path))
